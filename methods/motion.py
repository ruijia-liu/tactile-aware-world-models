"""Check recorded actions, materials, and the physical outcome of one episode."""

import csv
import hashlib
import numpy as np

from simulation.planner import protected_grip_candidates

def require(ok, message):
    if not ok:
        raise ValueError(message)


def conj(q):
    return np.concatenate((q[..., :1], -q[..., 1:]), axis=-1)


def in_gripper_frame(plug, ee, ee_quaternion):
    """Plug centre relative to right EE, including rotation of the EE reference frame."""
    offset = plug - ee
    inv = conj(ee_quaternion)
    u = inv[:, 1:]
    v = 2 * np.cross(u, offset)
    return offset + inv[:, :1] * v + np.cross(u, v)


def normalized(q):
    q = q.astype(np.float64)
    length = np.linalg.norm(q, axis=-1, keepdims=True)
    require(np.allclose(length, 1, atol=1e-3, rtol=0), "invalid unit wxyz quaternions")
    return q / length


def score(row, raw, design):
    path=raw/(row["episode_id"]+".npz")
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row["npz_sha256"]
    with np.load(path, allow_pickle=False) as z:
        state=z["state"].astype(float)
        ee=z["ee_position_w"][:,1].astype(float)
        command=z["action"]
        times=z["time_s"]
        pads=z["tactile"][:,2:4]
        ui=z["ui_action"]
        phases=z["phase"]
        k,j,stop=(row[key] for key in ("first_lift_index","decision_index","release_index"))
        assert state.shape==(len(phases)+1,71) and command.shape==(len(phases),16)
        assert pads.shape==(len(state),2,12,32) and np.isfinite(pads).all() and pads.min()>=0 and pads.max()<=1
        assert np.isfinite(state).all() and np.isfinite(ee).all() and np.isfinite(command).all() and np.isfinite(ui).all()
        assert k==np.flatnonzero(phases=="lift")[0] and j==k+20 and stop==np.flatnonzero(phases=="release")[0]
        assert hashlib.sha256(np.ascontiguousarray(command[:j]).tobytes()).hexdigest()==row["prefix_sha256"]
        assert row["material_readback_start"]==row["material_readback_end"]
        for name,expected in zip(row["material_readback_start"], (design["plug_materials"][row["plug_material"]],
                              design["pad_materials"][row["pad_material"]])):
            np.testing.assert_allclose(np.array(name["shape_values"])[:,:2],
                                       np.broadcast_to(expected, (len(name["shape_values"]), 2)),atol=1e-5,rtol=0)
        np.testing.assert_allclose(np.diff(times),1/design["native_hz"],rtol=1e-5,atol=1e-8)
        pos=state[:,45:48]
        rel=in_gripper_frame(pos,ee,normalized(z["ee_quaternion_w"][:,1]))
        grip_begin = int(np.flatnonzero(phases=="grip")[0])
        prior_hits=pads[grip_begin:k+1].max(axis=(1,2,3))>=.01
        prior_contact=bool(len(prior_hits)>=3 and np.lib.stride_tricks.sliding_window_view(prior_hits,3).all(axis=1).any())
        gripper_excursion=float(1000*np.linalg.norm(ee[k:stop+1]-ee[k],axis=1).max())
        rel_distance=np.linalg.norm(rel[k:stop+1]-np.median(rel[k-11:k+1],axis=0),axis=1)*1000
        rolling=np.median(np.lib.stride_tricks.sliding_window_view(rel_distance,6),axis=-1)
        crossings=np.flatnonzero(rolling>=8)
        onset=int(crossings[0]+5) if len(crossings) else None
        cfg=design["success"]
        endpoint=min(stop,k+cfg["deadline_steps_from_lift"])
        goal=np.array(row["target_xyz_m"])
        errors=np.linalg.norm(pos[k:endpoint+1]-goal,axis=1)*1000
        heights=(pos[k:endpoint+1,2]-pos[k,2])*1000
        near=(errors<=cfg["goal_distance_mm"]) & (heights>=cfg["lift_mm"])
        completed=np.flatnonzero(np.lib.stride_tricks.sliding_window_view(near,cfg["hold_steps"]+1).all(axis=1))
        time_to_complete=(int(completed[0])+cfg["hold_steps"])/design["native_hz"] if len(completed) else None
        anchor=np.r_[pos[j],state[j,52:55],ee[j],state[j,14]-state[j,15]]
        k_anchor=np.r_[pos[k],state[k,52:55],ee[k],state[k,14]-state[k,15]]
        result=dict(episode_id=row["episode_id"], scenario=row["scenario_id"],family=row["family_id"],arm=row["candidate"],
                    plug=row["plug_material"],pad=row["pad_material"],steps_to_release=stop-k,
                    success=int(time_to_complete is not None), time_to_complete_s=time_to_complete,
                    terminal_hold=int(len(completed)>0 and int(completed[-1])+cfg["hold_steps"]==len(near)-1),
                    slip_h=onset, slip_before_release=int(onset is not None),
                    slip_h60=int(onset is not None and onset<=60),
                    peak_sustained_drift_mm=float(rolling.max()),terminal_error_mm=float(errors[-1]),
                    peak_lift_mm=float(heights.max()), minimum_aperture_mm=float(1000*np.min(state[k:stop+1,14]-state[k:stop+1,15])),
                    minimum_command_fraction=float(np.min(ui[k:stop,15])),
                    prior_contact=int(prior_contact),gripper_excursion_mm=gripper_excursion,
                    slip_qc_pass=int(prior_contact and gripper_excursion>=10))
        if "controller" in row:
            ds=row["decisions"]
            expected=np.arange(j-k,stop-k,4)
            assert np.array_equal([d["step_from_lift"] for d in ds],expected)
            assert [d["object_observed"] for d in ds]==[not(row["blackout"] and h<=20) for h in expected]
            # Verify the executed finger setpoint against the chosen candidate at every applied control step.
            for d in ds:
                t = k+d["step_from_lift"]
                count=min(4,stop-t)
                limits = protected_grip_candidates(phases, t, count, design, terminal=row.get("protect_terminal_hold"),
                                                   transport=row.get("protect_transport_hold"))
                if limits is not None:
                    assert d["eligible_candidates"] == limits
                else:
                    assert "eligible_candidates" not in d
                    limits = range(len(design["branches"]))
                assert d["selected"] == min(limits, key=d["objective"].__getitem__)
                fraction=list(design["branches"].values())[d["selected"]]
                delta=design["maximum_opening_change_per_step"]*np.arange(1,count+1)
                prior=ui[t-1,15]
                chosen=prior+np.clip(fraction-prior,-delta,delta)
                np.testing.assert_allclose(ui[t:t+count,15],chosen,rtol=0,atol=1e-7)
                np.testing.assert_allclose(np.array(d["mean_cost"])+row["risk_weight"]*np.array(d["sd_cost"]),
                                           d["objective"],rtol=1e-9,atol=1e-9)
            if row.get("protect_transport_hold"):
                assert int(np.count_nonzero(phases == "hold")) == int(np.ceil(row["hold_seconds"] / row["physics_dt"]))
                holding = np.isin(phases, ("transport", "hold"))
                assert float(ui[holding, 15].max()) <= design["reference_gripper"] + 1e-7
            result.update(selections=len(ds),choices_tight=sum(d["selected"]==0 for d in ds),
                          switches=int(np.count_nonzero(np.diff([d["selected"] for d in ds]))))
        else:
            bound=design["maximum_opening_change_per_step"]
            previous=ui[j-1,15]
            for t in range(j,stop):
                previous=float(np.clip(design["branches"][row["candidate"]],previous-bound,previous+bound))
                np.testing.assert_allclose(ui[t,15],previous,rtol=0,atol=1e-7)
        np.testing.assert_allclose(command[j:stop,14],.042*ui[j:stop,15],atol=1e-7,rtol=0)
        np.testing.assert_allclose(command[j:stop,15],-.042*ui[j:stop,15],atol=1e-7,rtol=0)
        return result, {"k":k_anchor,"j":anchor,"goal":goal,"prefix":row["prefix_sha256"],
                        "ready_prefix":row["common_ready_prefix_sha256"]}


def offsets(first,second):
    return dict(max_plug_distance_mm=float(np.linalg.norm(first[:3]-second[:3])*1000),
                max_plug_velocity_difference_mm_s=float(np.linalg.norm(first[3:6]-second[3:6])*1000),
                max_right_ee_distance_mm=float(np.linalg.norm(first[6:9]-second[6:9])*1000),
                max_aperture_difference_mm=float(abs(first[9]-second[9])*1000))


def save_rows(path,rows):
    if rows:
        with path.open("w",newline="",encoding="utf8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

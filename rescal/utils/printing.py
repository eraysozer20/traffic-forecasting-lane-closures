# utils/printing.py

def print_test_results_block(
    mae_b, rmse_b, mape_b,
    mae_f, rmse_f, mape_f,
    split_name="C",
):
    """
    Prints:
    ========= TEST RESULTS (C) =========
    [BASELINE ] MAE=... RMSE=... MAPE=...
    [BASE+Res ] MAE=... RMSE=... MAPE=...
    ===================================
    """
    print(f"\n========== TEST RESULTS ({split_name}) ==========")
    print(f"[BASELINE ] MAE={mae_b:.4f} RMSE={rmse_b:.4f} MAPE={mape_b:.4f}")
    print(f"[BASE+Res ] MAE={mae_f:.4f} RMSE={rmse_f:.4f} MAPE={mape_f:.4f}")
    print("=====================================\n")


def print_event_metrics_block(title: str, event_res: dict):
    """
    Prints event metrics in the exact format you requested.
    event_res must be the output of event_closure_metrics_from_arrays.
    """
    print(f"\n========== {title} ==========")

    print(
        f"[INFO] seen sensors: {event_res['seen_sensors_count']}, "
        f"unseen sensors: {event_res['unseen_sensors_count']}, "
        f"radius={event_res['radius_miles']} miles"
    )

    seen = event_res["seen"]
    unseen = event_res["unseen"]

    print(
        f"[SEEN]   events(all)={seen['events_all']}, "
        f"MAE_all={seen['MAE_all']:.4f}, RMSE_all={seen['RMSE_all']:.4f}"
    )
    print(
        f"[SEEN]   events(loc)={seen['events_local']}, "
        f"MAE_loc={seen['MAE_local']:.4f}, RMSE_loc={seen['RMSE_local']:.4f}"
    )
    print(
        f"[UNSEEN] events(all)={unseen['events_all']}, "
        f"MAE_all={unseen['MAE_all']:.4f}, RMSE_all={unseen['RMSE_all']:.4f}"
    )
    print(
        f"[UNSEEN] events(loc)={unseen['events_local']}, "
        f"MAE_loc={unseen['MAE_local']:.4f}, RMSE_loc={unseen['RMSE_local']:.4f}"
    )

    print("=====================================\n")

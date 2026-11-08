import sys
from pathlib import Path
import numpy as np
import pandas as pd

RENAME = {
    "dur": "flow_duration", "spkts": "src_packets", "dpkts": "dst_packets",
    "sbytes": "src_bytes", "dbytes": "dst_bytes", "rate": "flow_rate",
    "sttl": "src_ttl", "dttl": "dst_ttl", "sload": "src_load_bps", "dload": "dst_load_bps",
    "sloss": "src_loss", "dloss": "dst_loss", "sinpkt": "src_interpacket_time",
    "dinpkt": "dst_interpacket_time", "sjit": "src_jitter", "djit": "dst_jitter",
    "swin": "src_tcp_window", "dwin": "dst_tcp_window", "stcpb": "src_tcp_base_seq",
    "dtcpb": "dst_tcp_base_seq", "tcprtt": "tcp_rtt", "synack": "tcp_synack_time",
    "ackdat": "tcp_ackdat_time", "smean": "src_mean_pkt_size", "dmean": "dst_mean_pkt_size",
    "trans_depth": "http_trans_depth", "response_body_len": "http_response_len",
    "ct_srv_src": "conn_srv_src_count", "ct_state_ttl": "conn_state_ttl_count",
    "ct_dst_ltm": "conn_dst_lifetime", "ct_src_dport_ltm": "conn_src_dport_time",
    "ct_dst_sport_ltm": "conn_dst_sport_time", "ct_dst_src_ltm": "conn_dst_src_time",
    "is_ftp_login": "ftp_login_flag", "ct_ftp_cmd": "ftp_cmd_count",
    "ct_flw_http_mthd": "http_method_count", "ct_src_ltm": "conn_src_lifetime",
    "ct_srv_dst": "conn_srv_dst_count", "is_sm_ips_ports": "same_ip_port_flag",
}
FEATURES = list(RENAME.values())
assert len(FEATURES) == 39
UNSEEN_L3 = ["Analysis", "Shellcode", "Worms"]
N_TRAIN, N_TEST = 141_700, 38_650


def main(src_dir: Path, out_dir: Path):
    pool = pd.concat([pd.read_csv(src_dir / f, encoding="utf-8-sig")
                      for f in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")])
    pool["attack_cat"] = pool["attack_cat"].str.strip()
    pool = pool.rename(columns=RENAME)[FEATURES + ["attack_cat", "label"]].reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for level, seed in ((1, 11), (2, 22), (3, 33)):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(pool))[: N_TRAIN + N_TEST]
        tr, te = pool.iloc[idx[:N_TRAIN]].copy(), pool.iloc[idx[N_TRAIN:]].copy()
        if level == 3:
            tr = tr[~tr.attack_cat.isin(UNSEEN_L3)]
            te.loc[te.attack_cat.isin(UNSEEN_L3), "attack_cat"] = "Unknown"
        target = "label" if level == 1 else "attack_cat"
        tr.insert(0, "id", np.arange(len(tr)))
        te.insert(0, "id", np.arange(len(te)))
        tr[["id"] + FEATURES + [target]].to_csv(out_dir / f"level{level}_train.csv", index=False)
        te[["id"] + FEATURES].to_csv(out_dir / f"level{level}_test.csv", index=False)
        te[["id", target]].to_csv(out_dir / f"level{level}_test_labels.csv", index=False)
        print(f"level{level}: train {len(tr)}  test {len(te)}  "
              f"classes {tr[target].nunique()} -> test {te[target].value_counts().to_dict() if level==3 else te[target].nunique()}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(root / "downloads/unsw", Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data_mock")

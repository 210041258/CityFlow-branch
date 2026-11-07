#!/usr/bin/env python3
"""
powerful_datagram_generator_v3.py
==================================
مولّد بيانات V2X صناعية: بيئة طبيعية + تشفير + هجمات + IDS + مكافآت RL + ميزات مشتقة.

الاستخدام:
    python powerful_datagram_generator_v3.py --preset stress-test --export all --out stress_v31
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


# ============================================================
# 1) الإعدادات الجاهزة (Presets)
# ============================================================
PRESETS: dict[str, dict[str, Any]] = {
    "default":     {"n": 20_000,  "attack_rate": 0.05, "seed": 42,   "dt": 0.1},
    "quick":       {"n": 1_000,   "attack_rate": 0.05, "seed": 1,    "dt": 0.1},
    "stress-test": {"n": 500_000, "attack_rate": 0.47, "seed": 1337, "dt": 0.1},
    "demo":        {"n": 5_000,   "attack_rate": 0.20, "seed": 7,    "dt": 0.1},
}


# ============================================================
# 2) البيئة الطبيعية
# ============================================================
NATURAL_CONDITIONS: dict[str, dict[str, Any]] = {
    "N-clear":    {"snr": 20.0, "rain": 0.0,  "vis": 10000, "temp": 20},
    "N-rain":     {"snr": 12.0, "rain": 12.0, "vis": 2000,  "temp": 15},
    "N-snow":     {"snr":  8.0, "rain":  5.0, "vis": 500,   "temp": -2},
    "N-fog":      {"snr": 10.0, "rain":  0.0, "vis": 200,   "temp":  8},
    "N-accident": {"snr": 10.0, "rain":  0.0, "vis": 5000,  "temp": 18},
}

FADING_MODELS = ["Rician", "Rayleigh", "Nakagami"]

# ============================================================
# 3) رسائل V2X + طرق التشفير + البروتوكولات
# ============================================================
MESSAGE_TYPES: dict[str, dict[str, Any]] = {
    "BSM":  {"priority": 1, "base_bytes": 39,  "crypto": True},
    "CAM":  {"priority": 1, "base_bytes": 50,  "crypto": True},
    "DENM": {"priority": 1, "base_bytes": 80,  "crypto": True},
    "SPaT": {"priority": 2, "base_bytes": 130, "crypto": True},
    "MAP":  {"priority": 2, "base_bytes": 500, "crypto": True},
    "TIM":  {"priority": 3, "base_bytes": 250, "crypto": True},
}

# (الاسم، حجم المفتاح، مقاوم كمي، FLOPs لكل بايت)
ENCRYPTION_METHODS = [
    ("ECC-P256-ECDSA", 256, False, 16000.0),
    ("AES-128-GCM",    128, False,  1000.0),
    ("AES-256-GCM",    256, False,  2000.0),
    (None,             128, False,     0.0),   # بدون تشفير (20% من الحالات)
]

PROTOCOLS = ["DSRC-802.11p", "LTE-V2X", "C-V2X-PC5", "5G-NR-V2X"]

# ============================================================
# 4) جدول الهجمات
# ============================================================
ATTACK_TYPES: dict[str, dict[str, Any]] = {
    "wormhole":     {"severity": 5, "target": "combined",     "burst": (5, 20)},
    "replay_burst": {"severity": 4, "target": "nonce_replay", "burst": (4, 12)},
    "sybil_flood":  {"severity": 5, "target": "rate_anomaly", "burst": (8, 25)},
    "jamming":      {"severity": 5, "target": "weak_signal",  "burst": (6, 20)},
    "dos":          {"severity": 5, "target": "rate_anomaly", "burst": (3, 15)},
    "tamper":       {"severity": 5, "target": "combined",     "burst": (5, 20)},
    "blackhole":    {"severity": 5, "target": "combined",     "burst": (5, 18)},
    "sybil":        {"severity": 4, "target": "rate_anomaly", "burst": (4, 15)},
    "flood_syn":    {"severity": 4, "target": "weak_signal",  "burst": (5, 18)},
    "mitm":         {"severity": 5, "target": "combined",     "burst": (6, 22)},
    "grayhole":     {"severity": 4, "target": "combined",     "burst": (5, 20)},
    "slow_loris":   {"severity": 2, "target": "weak_signal",  "burst": (4, 16)},
}


def _hex(n: int) -> str:
    return secrets.token_hex(n)


# ============================================================
# 5) المولّد الرئيسي
# ============================================================
class DatagramGenerator:
    def __init__(self, n: int, attack_rate: float, seed: int = 42, dt: float = 0.1):
        self.n = n
        self.attack_rate = attack_rate
        self.dt = dt
        self.rng = __import__("random").Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # عدّادات تراكمية
        self.crypto_idx = 0
        self.key_rotation_idx = 0
        self.nonce_window_idx = 0
        self.session_idx = 0
        self.last_rotation_step = 0

        # تتبع الحملات
        self.campaigns = self._plan_campaigns()

        # تسلسل
        self.expected_seq = 1
        self.last_seq = 0
        self.last_seq_gap = 0
        self.max_seq_gap = 0
        self.cum_seq_gap = 0

        # نوافذ
        self.seen_nonces: set[str] = set()

        # خطوات أخيرة
        self.last_attack_step: int | None = None
        self.last_ids_step: int | None = None
        self.last_invalid_step: int | None = None
        self.last_switch_step = 0
        self.last_attack_end: int | None = None

        # سلاسل
        self.consec_invalid = 0
        self.consec_valid = 0
        self.consec_attack = 0
        self.longest_invalid = 0
        self.longest_valid = 0

        # تراكميات
        self.cum_attacks = 0
        self.cum_invalid = 0
        self.cum_alerts = 0

        # شلال
        self.cascade_active = False
        self.cascade_steps = 0

        # EWMA
        self.ewma_attack_recency = 0.0
        self.ewma_invalid_recency = 0.0
        self.ewma_interarrival = 0.0
        self.ewma_rotation = 0.0

        # صحة التشفير
        self.crypto_health_hist: list[float] = []

        # تراكمي backoff
        self.attack_backoff_count = 0

    # -------- تخطيط الحملات --------
    def _plan_campaigns(self) -> list[tuple[int, int, str, int]]:
        plan = []
        step = 1
        cid = 0
        while step <= self.n:
            if self.rng.random() < self.attack_rate:
                atype = self.rng.choice(list(ATTACK_TYPES.keys()))
                lo, hi = ATTACK_TYPES[atype]["burst"]
                length = self.rng.randint(lo, hi)
                end = min(self.n, step + length - 1)
                plan.append((step, end, atype, cid))
                cid += 1
                step = end + 1
            else:
                step += 1
        return plan

    def _campaign_at(self, step: int) -> tuple[bool, str, int, int]:
        for (s, e, atype, cid) in self.campaigns:
            if s <= step <= e:
                return True, atype, cid, step - s
        return False, "none", -1, 0

    # -------- اختيار التشفير --------
    def _pick_crypto(self, force_secure: bool):
        if not force_secure and self.rng.random() < 0.20:
            return ENCRYPTION_METHODS[3]  # بدون تشفير
        return self.rng.choice(ENCRYPTION_METHODS[:3])

    # -------- المكافأة --------
    @staticmethod
    def _reward(valid: bool, alert: bool, throughput: float,
                queue: float, spill: int,
                safety: float, efficiency: float) -> float:
        r = 0.0
        r += 2.0 if valid else -2.0
        r += -1.0 if alert else 0.0
        r += 0.02 * throughput
        r += -0.01 * queue
        r += -5.0 * spill
        r += safety + efficiency
        return round(r, 3)

    # -------- الحلقة الرئيسية --------
    def generate(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for i in range(1, self.n + 1):
            step = i
            t = round(i * self.dt, 3)

            # ---- الحملة ----
            in_camp, atype, cid, atk_step = self._campaign_at(step)
            attack_injected = in_camp
            if attack_injected:
                self.cum_attacks += 1
                self.consec_attack += 1
                if self.last_attack_step is not None:
                    dt_att = step - self.last_attack_step
                    self.ewma_interarrival = 0.9 * self.ewma_interarrival + 0.1 * dt_att
                self.last_attack_step = step
                self.last_attack_end = None
            else:
                self.consec_attack = 0
                if self.last_attack_end is None and self.cum_attacks > 0:
                    self.last_attack_end = step

            # ---- البيئة ----
            natural = self.rng.choice(list(NATURAL_CONDITIONS.keys()))
            nc = NATURAL_CONDITIONS[natural]
            snr_db = nc["snr"] + self.rng.gauss(0, 2)
            sinr_db = snr_db - self.rng.uniform(0, 5)

            if attack_injected:
                if atype in ("jamming", "wormhole"):
                    sinr_db -= self.rng.uniform(3, 12)
                    snr_db  -= self.rng.uniform(3, 12)
                elif atype == "tamper":
                    sinr_db -= self.rng.uniform(0, 3)
                elif atype == "mitm":
                    sinr_db -= self.rng.uniform(1, 5)

            snr_db, sinr_db = round(snr_db, 3), round(sinr_db, 3)

            # ---- الرسالة ----
            mtype = self.rng.choice(list(MESSAGE_TYPES.keys()))
            m = MESSAGE_TYPES[mtype]
            priority = m["priority"]
            payload = m["base_bytes"] + self.rng.randint(-10, 10)

            # ---- التشفير ----
            enc_name, key_bits, qr, flops_per_byte = self._pick_crypto(
                force_secure=not attack_injected
            )
            self.crypto_idx += 1

            # دوران المفاتيح كل 100 خطوة
            if step - self.last_rotation_step >= 100:
                self.key_rotation_idx += 1
                self.last_rotation_step = step

            if step % 10 == 0:
                self.nonce_window_idx += 1
            if step % 50 == 0:
                self.session_idx += 1

            # ---- Nonce ----
            reuse_nonce = (
                attack_injected
                and atype == "replay_burst"
                and self.seen_nonces
                and self.rng.random() < 0.7
            )
            if reuse_nonce:
                nonce = self.rng.choice(list(self.seen_nonces))
            else:
                nonce = _hex(6)

            nonce_reuse = nonce in self.seen_nonces
            self.seen_nonces.add(nonce)

            # ---- التسلسل ----
            seq = self.expected_seq
            if attack_injected and atype in ("replay_burst", "wormhole") \
                    and self.rng.random() < 0.5:
                seq += self.rng.randint(1, 5)
            gap_mag = 0
            if i > 1:
                gap_mag = max(0, abs(seq - self.last_seq) - 1)
            self.last_seq = seq
            self.expected_seq = seq + 1

            seq_gap_detected = gap_mag > 0
            if seq_gap_detected:
                self.cum_seq_gap += gap_mag
                self.max_seq_gap = max(self.max_seq_gap, gap_mag)
                self.last_seq_gap = gap_mag

            # ---- التوقيع ----
            sig_valid = True
            sig_fail_reason = "none"
            if attack_injected and atype in ("tamper", "wormhole", "blackhole", "mitm", "grayhole") \
                    and self.rng.random() < 0.9:
                sig_valid = False
                sig_fail_reason = "tampered"
            sig_hmac = _hex(8)

            # ---- الكمون ----
            if enc_name:
                base = flops_per_byte * payload
                crypto_lat = (base / 1e6) * self.rng.uniform(0.5, 1.5)
                if attack_injected and atype == "dos":
                    crypto_lat *= 3
            else:
                crypto_lat = 0.0

            # ---- القناة ----
            sinr_lin = 10 ** (sinr_db / 10)
            packet_err = 1.0 / (1.0 + sinr_lin) if sinr_lin > 0 else 1.0
            packet_err = round(
                min(0.999, max(0.0, packet_err + self.rng.uniform(-0.05, 0.05))), 6
            )

            ecc_strength = self.rng.choice([1, 2])
            ecc_ok = self.rng.random() > packet_err

            retrans = 0
            if not ecc_ok and packet_err > 0.5:
                retrans = self.rng.randint(1, 3)

            residual_ber = round(
                packet_err * (0.5 ** ecc_strength) if ecc_ok else packet_err, 6
            )

            # ---- الصلاحية ----
            valid = (
                sig_valid
                and ecc_ok
                and not nonce_reuse
                and not (attack_injected and atype in ("jamming", "dos"))
            )
            if attack_injected and atype in (
                "wormhole", "blackhole", "tamper",
                "mitm", "grayhole", "replay_burst",
            ) and self.rng.random() < 0.6:
                valid = False

            if valid:
                self.consec_valid += 1
                self.consec_invalid = 0
                self.longest_valid = max(self.longest_valid, self.consec_valid)
            else:
                self.cum_invalid += 1
                self.consec_invalid += 1
                self.consec_valid = 0
                self.longest_invalid = max(self.longest_invalid, self.consec_invalid)
                self.last_invalid_step = step

            # ---- IDS ----
            ids_alert = 0
            ids_reason = "none"
            intrusion = False
            if attack_injected:
                ids_reason = ATTACK_TYPES[atype]["target"]
                if self.rng.random() < 0.95:
                    ids_alert = 1
                    intrusion = True
                    self.cum_alerts += 1
                    self.last_ids_step = step
            elif nonce_reuse or seq_gap_detected or not sig_valid:
                ids_alert = 1
                ids_reason = "combined"
                intrusion = True
                self.cum_alerts += 1

            if ids_alert and self.rng.random() < 0.5:
                self.attack_backoff_count += 1

            # ---- الشلال ----
            if (not valid) and ids_alert:
                self.cascade_active = True
                self.cascade_steps += 1
            elif self.cascade_active and self.rng.random() < 0.3:
                self.cascade_active = False

            # ---- الطوابير والإنتاجية ----
            total_queue = max(
                0.0,
                450
                + 40 * math.sin(step / 10)
                + 30 * (1 if attack_injected else 0),
            )
            queues = [
                round(total_queue / 30 + self.rng.uniform(-5, 5), 2)
                for _ in range(30)
            ]
            phases = [0] * 30
            queue_var = round(float(np.var(queues)), 3)
            spill = 0
            throughput = round(
                60 + 20 * self.rng.random()
                - 50 * 0.5 * (1 if attack_injected else 0), 2
            )
            avg_speed = round(45 + self.rng.uniform(-3, 3), 2)

            switches = self.rng.randint(1, 5) if attack_injected else 0
            if switches:
                self.last_switch_step = step

            chan_load = round(
                min(1.0, 0.2 + 0.5 * (1 if attack_injected else 0)
                    + self.rng.uniform(-0.05, 0.05)), 4
            )

            # ---- الإجراء ----
            action = "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
            if ids_alert:
                action = "[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"

            dq_flags = 0
            if nonce_reuse:         dq_flags |= 1
            if seq_gap_detected:    dq_flags |= 2
            if not sig_valid:       dq_flags |= 4
            if not ecc_ok:          dq_flags |= 8

            reward_safety = round(
                0.1 * (1 if valid else -1) + 0.05 * (1 if ids_alert else 0), 4
            )
            reward_efficiency = round(
                0.001 * throughput - 0.0001 * total_queue, 4
            )
            reward = self._reward(
                valid, ids_alert, throughput, total_queue, spill,
                reward_safety, reward_efficiency
            )

            # ---- صحة التشفير ----
            self.crypto_health_hist.append(1.0 if valid else 0.0)
            if len(self.crypto_health_hist) > 10:
                self.crypto_health_hist.pop(0)
            crypto_health = round(float(np.mean(self.crypto_health_hist)), 4)

            # ---- EWMA ----
            if self.last_attack_step is not None:
                self.ewma_attack_recency = (
                    0.9 * self.ewma_attack_recency
                    + 0.1 * (step - self.last_attack_step)
                )
            if self.last_invalid_step is not None:
                self.ewma_invalid_recency = (
                    0.9 * self.ewma_invalid_recency
                    + 0.1 * (step - self.last_invalid_step)
                )
            self.ewma_rotation = round(0.9 * self.ewma_rotation + 0.1 * 100, 2)

            # ---- النسب ----
            lat_ratio   = round(crypto_lat / 50.0, 6)
            q_cap_ratio = round(total_queue / 500.0, 6)
            defense_r   = round(
                1.0 if (valid and ids_alert) else 0.5 if valid else 0.0, 3
            )
            thr_ratio   = round(throughput / 100.0, 4)
            coding_gain = round(max(0.0, 1.0 - residual_ber), 4)
            key_fresh   = round(1.0 - (step - self.last_rotation_step) / 100.0, 4)

            # ---- المقاطع ----
            def seg_lat(v):   return 0 if v < 0.10 else 1 if v < 0.30 else 2 if v < 0.60 else 3
            def seg_cong(v):  return 0 if v < 0.25 else 1 if v < 0.50 else 2 if v < 0.75 else 3
            def seg_phase(s): return 0 if s < 30 else 1 if s < 60 else 2 if s < 120 else 3
            def seg_sinr(v):  return 0 if v < 0 else 1 if v < 5 else 2 if v < 10 else 3

            # ---- Bits / Risks ----
            perm_bits = max(1, int(math.log2(self.cum_attacks + 1))) * 4
            comb_bits = max(1, int(math.log2(len(ATTACK_TYPES) + 1))) * 2
            msg_bits  = max(1, int(math.log2(len(MESSAGE_TYPES) + 1))) * 2
            nonce_birthday = (len(self.seen_nonces) ** 2) / (2 ** 48)
            nonce_birthday_risk = round(min(1.0, nonce_birthday), 6)
            key_epoch_factor = round((step - self.last_rotation_step) / 100.0, 4)
            nonce_sat = round(len(self.seen_nonces) / 10_000.0, 4)

            # ---- بناء السجل ----
            row = {
                "Datagram_Index": i,
                "Step": step,
                "Sim_Time_s": t,
                "Natural_Condition": natural,
                "SNR_dB": snr_db,
                "SINR_dB": sinr_db,
                "Rain_Rate_mm_h": nc["rain"],
                "Visibility_m": nc["vis"],
                "Temperature_C": nc["temp"],
                "Fading_Model": self.rng.choice(FADING_MODELS),
                "Doppler_Hz": round(abs(self.rng.gauss(0, 300)) + 50, 2),
                "Shadowing_dB": round(self.rng.gauss(0, 3), 3),
                "Doppler_Coherence": round(self.rng.uniform(0.60, 0.95), 4),
                "Message_Type": mtype,
                "Message_Priority": priority,
                "Payload_Bytes": payload,
                "Encryption_Method": enc_name,
                "Key_Bits": key_bits,
                "Quantum_Resistant": int(qr),
                "Crypto_Latency_MS": round(crypto_lat, 4),
                "Crypto_Flops": flops_per_byte,
                "Crypto_Index": self.crypto_idx,
                "Key_Rotation_Index": self.key_rotation_idx,
                "Nonce_Window_Index": self.nonce_window_idx,
                "Session_Index": self.session_idx,
                "Sequence_Number": seq,
                "Signature_Failure_Reason": sig_fail_reason,
                "Nonce_Reuse_Detected": nonce_reuse,
                "Sequence_Gap_Detected": seq_gap_detected,
                "Nonce": nonce,
                "Signature_HMAC": sig_hmac,
                "Signature_Valid": sig_valid,
                "Attack_Index": i if attack_injected else 0,
                "Attack_Campaign_ID": cid,
                "Attack_Type": atype,
                "Attack_Target_RSU": self.rng.randint(-1, 30) if attack_injected else -1,
                "Attack_Step_In_Campaign": atk_step,
                "Attack_Escalation_Level": round(min(1.0, atk_step / 20.0), 3),
                "Attack_Injected": attack_injected,
                "Attack_Severity": ATTACK_TYPES[atype]["severity"] if attack_injected else 0,
                "IDS_Alert_Level": ids_alert,
                "IDS_Reason": ids_reason,
                "Intrusion_Detected": intrusion,
                "Protocol": self.rng.choice(PROTOCOLS),
                "RSU_ID": self.rng.randint(0, 30),
                "Channel_Load": chan_load,
                "Packet_Error_Prob": packet_err,
                "Retransmissions": retrans,
                "ECC_Strength": ecc_strength,
                "ECC_Success": ecc_ok,
                "Residual_BER": residual_ber,
                "Action_Taken": action,
                "Switches": switches,
                "Queues_Per_Intersection": json.dumps(queues),
                "Phases_Per_Intersection": json.dumps(phases),
                "Total_Queue": round(total_queue, 2),
                "Queue_Variance": queue_var,
                "Spilled_Vehicles": spill,
                "Vehicle_Throughput": throughput,
                "Avg_Speed_kmh": avg_speed,
                "Reward": reward,
                "Reward_Safety": reward_safety,
                "Reward_Efficiency": reward_efficiency,
                "Datagram_Valid": valid,
                "Data_Quality_Flags": dq_flags,
                "Sustained_Valid_Streak": self.consec_valid,
                "Crypto_Health_Rolling_10": crypto_health,
                "Attack_Backoff_Count": self.attack_backoff_count,
                "Permutation_Complexity_bits": perm_bits,
                "Attack_Combination_bits": comb_bits,
                "Message_Permutation_bits": msg_bits,
                "Nonce_Birthday_Risk": nonce_birthday_risk,
                "Key_Epoch_Factor": key_epoch_factor,
                "Nonce_Cache_Saturation": nonce_sat,
                "SINR_Linear_Ratio": round(10 ** (sinr_db / 10), 6),
                "Latency_Ratio": lat_ratio,
                "Queue_Capacity_Ratio": q_cap_ratio,
                "Attack_Defense_Ratio": defense_r,
                "Throughput_Arrival_Ratio": thr_ratio,
                "Coding_Gain_Ratio": coding_gain,
                "Key_Freshness_Ratio": key_fresh,
                "Latency_Segment": seg_lat(lat_ratio),
                "Congestion_Segment": seg_cong(q_cap_ratio),
                "Attack_Phase_Segment": seg_phase(atk_step),
                "SINR_Segment": seg_sinr(sinr_db),
                "Steps_Since_Last_Attack": (
                    step - self.last_attack_step if self.last_attack_step is not None else 0
                ),
                "Steps_Since_Last_IDS_Alert": (
                    step - self.last_ids_step if self.last_ids_step is not None else step
                ),
                "Steps_Since_Last_Switch": step - self.last_switch_step if switches else 0,
                "Steps_Since_Key_Rotation": step - self.last_rotation_step,
                "Consecutive_Invalid_Streak": self.consec_invalid,
                "Last_Sequence_Gap_Magnitude": self.last_seq_gap,
                "Failure_Cascade_Active": self.cascade_active,
                "Consecutive_Valid_Streak": self.consec_valid,
                "Consecutive_Attack_Streak": self.consec_attack,
                "Longest_Invalid_Streak": self.longest_invalid,
                "Longest_Valid_Streak": self.longest_valid,
                "Steps_Since_Attack_End": (
                    step - self.last_attack_end if self.last_attack_end is not None else 0
                ),
                "Steps_Since_Last_Invalid": (
                    step - self.last_invalid_step if self.last_invalid_step is not None else 0
                ),
                "Cumulative_Attack_Count": self.cum_attacks,
                "Cumulative_Invalid_Count": self.cum_invalid,
                "Cumulative_Alert_Count": self.cum_alerts,
                "Cumulative_Seq_Gap_Magnitude": self.cum_seq_gap,
                "Max_Sequence_Gap_Magnitude": self.max_seq_gap,
                "Attack_Density_Window": round(self.consec_attack / 30.0, 4),
                "Alert_Density_Window": round(ids_alert / 30.0, 4),
                "Invalid_Density_Window": round(self.consec_invalid / 30.0, 4),
                "EWMA_Interarrival_Attacks": round(self.ewma_interarrival, 4),
                "EWMA_Attack_Recency": round(self.ewma_attack_recency, 4),
                "EWMA_Invalid_Recency": round(self.ewma_invalid_recency, 4),
                "EWMA_Rotation_Interval": self.ewma_rotation,
                "Log_Steps_Since_Attack": round(math.log1p(
                    max(0, step - self.last_attack_step)
                    if self.last_attack_step is not None else 0
                ), 4),
                "Log_Steps_Since_Invalid": round(math.log1p(
                    max(0, step - self.last_invalid_step)
                    if self.last_invalid_step is not None else 0
                ), 4),
            }
            rows.append(row)

        return rows


# ============================================================
# 6) حساب الملخص
# ============================================================
def compute_summary(rows: list[dict[str, Any]], filename: str = "") -> dict[str, Any]:
    n = len(rows)
    valid       = sum(1 for r in rows if r["Datagram_Valid"])
    attacks     = sum(1 for r in rows if r["Attack_Injected"])
    alerts      = sum(1 for r in rows if r["IDS_Alert_Level"])
    nonce_rep   = sum(1 for r in rows if r["Nonce_Reuse_Detected"])
    seq_gaps    = sum(1 for r in rows if r["Sequence_Gap_Detected"])
    sig_fail    = sum(1 for r in rows if not r["Signature_Valid"])
    backoffs    = sum(1 for r in rows if r["Attack_Backoff_Count"])
    switches    = sum(r["Switches"] for r in rows)
    spilled     = sum(r["Spilled_Vehicles"] for r in rows)
    reward      = sum(r["Reward"] for r in rows) / n if n else 0.0
    final_queue = rows[-1]["Total_Queue"] if rows else 0.0

    cascade_steps = sum(1 for r in rows if r["Failure_Cascade_Active"])
    cascade_windows, prev = 0, False
    for r in rows:
        if r["Failure_Cascade_Active"] and not prev:
            cascade_windows += 1
        prev = r["Failure_Cascade_Active"]

    return {
        "File": filename,
        "Valid": f"{valid} ({valid / n * 100:.1f}%)" if n else "0",
        "Dropped": f"{n - valid} ({(n - valid) / n * 100:.1f}%)" if n else "0",
        "Attacks": attacks,
        "IDS alerts": alerts,
        "Nonce replays": nonce_rep,
        "Seq gaps": seq_gaps,
        "Sig failures": sig_fail,
        "Backoffs": backoffs,
        "Cascade windows": f"{cascade_windows} ({cascade_steps} steps in cascade)",
        "Total switches": switches,
        "Final queue": round(final_queue, 2),
        "Avg reward": round(reward, 2),
        "Total spilled": spilled,
    }


# ============================================================
# 7) دوال التصدير
# ============================================================
def export_csv(rows, path: str):
    if pd is None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def export_parquet(rows, path: str):
    if pd is None:
        print("  [skip] pandas غير مثبّت"); return
    try:
        pd.DataFrame(rows).to_parquet(path, index=False)
    except Exception as e:
        print(f"  [skip] parquet: {e}")


def export_jsonl(rows, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def export_npz(rows, path: str):
    arr = np.array([
        [float(v) if isinstance(v, (int, float)) else 0.0 for v in r.values()]
        for r in rows
    ])
    np.savez_compressed(path, data=arr)


def export_schema(rows, path: str):
    schema = {k: type(v).__name__ for k, v in rows[0].items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


def config_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(s).hexdigest()[:16]


# ============================================================
# 8) نقطة الدخول (CLI)
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="مولّد داتاغرام V2X صناعي مع هجمات و IDS"
    )
    ap.add_argument("--preset", choices=list(PRESETS), default="default")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--attack-rate", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--export", default="all",
                    help="csv | parquet | jsonl | npz | schema | all")
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.n is not None:            cfg["n"] = args.n
    if args.attack_rate is not None:  cfg["attack_rate"] = args.attack_rate
    if args.seed is not None:         cfg["seed"] = args.seed

    print(f"Generating {cfg['n']} datagrams -> {args.out}.csv")

    gen = DatagramGenerator(
        n=cfg["n"], attack_rate=cfg["attack_rate"],
        seed=cfg["seed"], dt=cfg["dt"],
    )
    rows = gen.generate()

    # تقدم تقريبي (للعرض فقط، لأن التوليد دفعة واحدة)
    chunk = 25_000
    for i in range(chunk, len(rows) + 1, chunk):
        print(f"  {i}/{len(rows)} ({i / len(rows) * 100:.1f}%)")

    targets = (["csv", "parquet", "jsonl", "npz", "schema"]
               if args.export == "all" else args.export.split(","))

    if "csv"     in targets: export_csv(rows, f"{args.out}.csv")
    if "parquet" in targets: export_parquet(rows, f"{args.out}.parquet")
    if "jsonl"   in targets: export_jsonl(rows, f"{args.out}.jsonl")
    if "npz"     in targets: export_npz(rows, f"{args.out}.npz")
    if "schema"  in targets: export_schema(rows, f"{args.out}.schema.json")

    summary = compute_summary(rows, filename=f"{args.out}.csv")
    summary["Config hash"] = config_hash(cfg)

    print("\n" + "=" * 60)
    for k, v in summary.items():
        print(f"  {k:18s} {v}")
    print("=" * 60)

    meta_path = f"{args.out}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
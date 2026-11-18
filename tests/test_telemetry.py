from datetime import datetime, timedelta

import pytest

from parentheses.telemetry import main, read_rows, summarise

HEADER = ("timestamp, power.draw [W], temperature.gpu, utilization.gpu [%], memory.used [MiB], "
          "clocks.current.sm [MHz], clocks_event_reasons.active\n")


def write_log(path, seconds, watts=100.0, step=1, skip=(), reasons="0x0000000000000000"):
    t0 = datetime(2026, 11, 18, 16, 0, 0)
    lines = [HEADER]
    for s in range(0, seconds + 1, step):
        if s in skip:
            continue
        ts = (t0 + timedelta(seconds=s)).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        lines.append(f"{ts}, {watts}, 70, 99, 7000, 2500, {reasons}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_constant_power_hour_is_a_tenth_kwh(tmp_path):
    out = summarise(read_rows(write_log(tmp_path / "a.csv", 3600)))
    assert out["gpu_kwh"] == pytest.approx(0.1)
    assert out["hours_logged"] == pytest.approx(1.0)
    assert out["hours_gap"] == 0


def test_gap_is_not_bridged(tmp_path):
    # 10 minutes missing in the middle of an hour: that time adds no energy.
    out = summarise(read_rows(write_log(tmp_path / "a.csv", 3600, skip=range(1201, 1800))))
    assert out["hours_gap"] == pytest.approx(600 / 3600)
    assert out["gpu_kwh"] == pytest.approx(0.1 * 3000 / 3600)


def test_unreadable_power_and_cut_line_are_skipped(tmp_path):
    p = write_log(tmp_path / "a.csv", 10)
    p.write_text(p.read_text().replace(", 100.0,", ", [N/A],", 1) + "2026/11/18 16:00:1", encoding="utf-8")
    out = summarise(read_rows(p))
    assert out["samples"] == 11
    assert out["gpu_kwh"] == pytest.approx(100 * 9 / 3.6e6)


def test_thermal_throttle_share(tmp_path):
    out = summarise(read_rows(write_log(tmp_path / "a.csv", 100, reasons="0x0000000000000040")))
    assert out["share_thermal_throttled"] == pytest.approx(1.0)
    assert out["share_power_capped"] == 0


def test_carbon_needs_a_source(tmp_path):
    p = write_log(tmp_path / "a.csv", 60)
    with pytest.raises(SystemExit):
        main([str(p), "--grid-kg-per-kwh", "0.7"])
    assert main([str(p), "--grid-kg-per-kwh", "0.7", "--grid-source", "test", "--tokens", "1000000"]) == 0

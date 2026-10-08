"""Kiểm tra tiến trình chấm điểm không yêu cầu giao diện đồ họa."""

from evaluate_practice import run_trackeval


def test_evaluation_disables_gui_plotting(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_mot_challenge.py").write_text(
        "import sys\n"
        "assert '--PLOT_CURVES' in sys.argv, 'Tiến trình chấm còn bật đồ thị GUI'\n"
        "assert sys.argv[sys.argv.index('--PLOT_CURVES') + 1] == 'False'\n"
        "assert sys.argv[sys.argv.index('--SEQ_INFO') + 1] == 'video_1'\n",
        encoding="utf-8",
    )
    run_trackeval(tmp_path, "Susois", "lab", "train")

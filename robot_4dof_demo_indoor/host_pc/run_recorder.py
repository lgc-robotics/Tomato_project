"""保存一次整机运行的控制台日志、结果图和图文报告。"""

import atexit
import html
import sys
import threading
from datetime import datetime
from pathlib import Path


class _TeeStream:
    """把输出同时写到原控制台和运行日志。"""

    def __init__(self, original_stream, recorder):
        self._original_stream = original_stream
        self._recorder = recorder

    def write(self, text):
        if not text:
            return 0

        written = self._original_stream.write(text)
        self._recorder.append_console_text(text)
        return len(text) if written is None else written

    def flush(self):
        self._original_stream.flush()
        self._recorder.flush_log()

    @property
    def encoding(self):
        return getattr(self._original_stream, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._original_stream, "errors", "strict")

    def isatty(self):
        return bool(getattr(self._original_stream, "isatty", lambda: False)())

    def __getattr__(self, name):
        return getattr(self._original_stream, name)


class RunRecorder:
    """管理一次运行产生的全部诊断资料。"""

    def __init__(self, project_dir, folder_name, max_log_chars_per_image):
        self.project_dir = Path(project_dir).resolve()
        self.root_dir = self.project_dir / folder_name
        self.run_started_at = datetime.now()
        self.run_dir = self._create_unique_run_dir()
        self.images_dir = self.run_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.depth_diagnostics_dir = self.run_dir / "depth_diagnostics"
        self.depth_diagnostics_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.run_dir / "控制台日志.txt"
        self.report_path = self.run_dir / "运行图文报告.html"
        self._log_file = self.log_path.open("w", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()
        self._pending_console_parts = []
        self._image_entries = []
        self._capture_labels = {}
        self._capture_counter = 0
        self._image_counter = 0
        self._depth_diagnostic_counter = 0
        self._finalized = False
        self.max_log_chars_per_image = max(
            1000,
            int(max_log_chars_per_image),
        )

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stdout_tee = _TeeStream(self.original_stdout, self)
        self.stderr_tee = _TeeStream(self.original_stderr, self)

    def _create_unique_run_dir(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self.run_started_at.strftime("%Y%m%d_%H%M%S")
        candidate = self.root_dir / f"运行_{timestamp}"
        suffix = 1

        while candidate.exists():
            candidate = self.root_dir / f"运行_{timestamp}_{suffix:02d}"
            suffix += 1

        candidate.mkdir(parents=True)
        return candidate

    def start_console_capture(self):
        sys.stdout = self.stdout_tee
        sys.stderr = self.stderr_tee

    def append_console_text(self, text):
        with self._lock:
            if self._finalized:
                return

            self._log_file.write(text)
            if "\n" in text:
                self._log_file.flush()
            self._pending_console_parts.append(text)

    def _limit_report_log_text(self, text):
        if len(text) <= self.max_log_chars_per_image:
            return text

        omitted_count = len(text) - self.max_log_chars_per_image
        return (
            f"[前面 {omitted_count} 个字符已省略，完整内容请查看控制台日志.txt]\n"
            + text[-self.max_log_chars_per_image:]
        )

    def flush_log(self):
        with self._lock:
            if not self._finalized:
                self._log_file.flush()

    def begin_capture_session(self, label):
        with self._lock:
            self._capture_counter += 1
            capture_id = self._capture_counter
            self._capture_labels[capture_id] = str(label or "视觉检测")
            return capture_id

    def allocate_image_path(self, capture_id, frame_id, image_kind="final"):
        with self._lock:
            self._image_counter += 1
            image_id = self._image_counter
            image_kind = str(image_kind or "final").strip().lower()
            if image_kind not in {"final", "threshold"}:
                image_kind = "result"
            filename = (
                f"image_{image_id:06d}_"
                f"session_{int(capture_id):04d}_frame_{int(frame_id) + 1:03d}_"
                f"{image_kind}.jpg"
            )
            return image_id, self.images_dir / filename

    def allocate_depth_diagnostic_paths(self, capture_id, frame_id, target_index):
        """为一帧中的一个果梗分配深度诊断图和原始数值文件。"""
        with self._lock:
            self._depth_diagnostic_counter += 1
            diagnostic_id = self._depth_diagnostic_counter
            stem = (
                f"depth_{diagnostic_id:06d}_"
                f"session_{int(capture_id):04d}_"
                f"frame_{int(frame_id) + 1:03d}_"
                f"target_{int(target_index) + 1:02d}"
            )
            return (
                self.depth_diagnostics_dir / f"{stem}.png",
                self.depth_diagnostics_dir / f"{stem}.npz",
            )

    def register_image(self, image_id, image_path, capture_id, frame_id, summary):
        with self._lock:
            console_excerpt = self._limit_report_log_text(
                "".join(self._pending_console_parts)
            )
            self._pending_console_parts.clear()
            relative_path = Path(image_path).resolve().relative_to(self.run_dir)
            self._image_entries.append({
                "image_id": int(image_id),
                "relative_path": relative_path.as_posix(),
                "capture_id": int(capture_id),
                "capture_label": self._capture_labels.get(
                    int(capture_id),
                    "视觉检测",
                ),
                "frame_number": int(frame_id) + 1,
                "summary": str(summary or ""),
                "console_excerpt": console_excerpt,
            })

    def _build_report_html(self, tail_console_text, finished_at):
        sections = []

        for entry in self._image_entries:
            sections.append(
                "\n".join([
                    f'<section id="image-{entry["image_id"]}">',
                    (
                        f'<h2>结果图 #{entry["image_id"]:06d}</h2>'
                        f'<p class="meta">检测批次 #{entry["capture_id"]:04d} · '
                        f'{html.escape(entry["capture_label"])} · '
                        f'第 {entry["frame_number"]} 帧</p>'
                    ),
                    (
                        f'<a href="{html.escape(entry["relative_path"])}">'
                        f'<img src="{html.escape(entry["relative_path"])}" '
                        f'alt="结果图 #{entry["image_id"]:06d}" loading="lazy"></a>'
                    ),
                    f'<p class="summary">{html.escape(entry["summary"])}</p>',
                    "<details>",
                    "<summary>查看这张图对应的控制台信息</summary>",
                    f'<pre>{html.escape(entry["console_excerpt"])}</pre>',
                    "</details>",
                    "</section>",
                ])
            )

        tail_section = ""
        if tail_console_text:
            tail_section = "\n".join([
                '<section id="tail-log">',
                "<h2>最后一张结果图之后的控制台信息</h2>",
                f"<pre>{html.escape(tail_console_text)}</pre>",
                "</section>",
            ])

        started_text = self.run_started_at.strftime("%Y-%m-%d %H:%M:%S")
        finished_text = finished_at.strftime("%Y-%m-%d %H:%M:%S")
        body = "\n".join(sections)

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>圣女果采摘运行图文报告</title>
  <style>
    body {{ margin: 0; color: #202124; background: #f5f6f7; font-family: "Microsoft YaHei", sans-serif; }}
    header {{ padding: 24px; color: white; background: #24543a; }}
    header h1 {{ margin: 0 0 8px; font-size: 24px; }}
    header p {{ margin: 4px 0; }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; }}
    nav {{ padding: 18px 0; }}
    nav a {{ color: #145da0; }}
    section {{ padding: 22px 0 28px; border-bottom: 1px solid #c8cdd2; }}
    h2 {{ margin: 0 0 6px; font-size: 20px; }}
    .meta {{ margin: 0 0 14px; color: #59636e; }}
    .summary {{ font-weight: 600; }}
    img {{ display: block; width: min(100%, 960px); height: auto; border: 1px solid #9da5ad; background: white; }}
    details {{ margin-top: 12px; }}
    summary {{ cursor: pointer; color: #145da0; }}
    pre {{ overflow-x: auto; padding: 14px; color: #e8eaed; background: #202124; white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>
  <header>
    <h1>圣女果采摘运行图文报告</h1>
    <p>开始时间：{html.escape(started_text)}</p>
    <p>结束时间：{html.escape(finished_text)}</p>
    <p>已保存结果图：{len(self._image_entries)} 张</p>
    <p>已保存果梗深度诊断：{self._depth_diagnostic_counter} 组</p>
  </header>
  <main>
    <nav><a href="控制台日志.txt">打开完整控制台日志</a> · <a href="depth_diagnostics/">打开果梗深度诊断目录</a></nav>
    {body}
    {tail_section}
  </main>
</body>
</html>
"""

    def finalize(self):
        report_error = None

        with self._lock:
            if self._finalized:
                return self.report_path

            try:
                finished_at = datetime.now()
                tail_console_text = self._limit_report_log_text(
                    "".join(self._pending_console_parts)
                )
                self._pending_console_parts.clear()
                report_html = self._build_report_html(
                    tail_console_text,
                    finished_at,
                )
                self.report_path.write_text(report_html, encoding="utf-8")
            except Exception as exc:
                report_error = exc
            finally:
                try:
                    self._log_file.flush()
                except OSError:
                    pass
                self._finalized = True

        if sys.stdout is self.stdout_tee:
            sys.stdout = self.original_stdout
        if sys.stderr is self.stderr_tee:
            sys.stderr = self.original_stderr

        try:
            self._log_file.close()
        except OSError:
            pass

        if report_error is not None:
            self.original_stderr.write(
                f"图文报告生成失败，控制台日志和结果图仍已保留: {report_error}\n"
            )
            self.original_stderr.flush()
            return None

        return self.report_path


_active_recorder = None


def start_run_recorder(
    project_dir,
    folder_name="运行记录",
    enabled=True,
    max_log_chars_per_image=30000,
):
    """启动运行记录；失败时不阻断机械臂主流程。"""
    global _active_recorder

    if not enabled:
        return None

    try:
        recorder = RunRecorder(
            project_dir,
            folder_name,
            max_log_chars_per_image,
        )
        recorder.start_console_capture()
        _active_recorder = recorder
        atexit.register(finalize_run_recorder)
        print(f"运行记录目录已创建: {recorder.run_dir}")
        return recorder.run_dir
    except Exception as exc:
        print(f"运行记录启动失败，程序继续运行: {exc}")
        _active_recorder = None
        return None


def begin_capture_session(label):
    if _active_recorder is None:
        return 0
    return _active_recorder.begin_capture_session(label)


def allocate_result_image_path(capture_id, frame_id, image_kind="final"):
    if _active_recorder is None:
        filename = f"{str(image_kind or 'final')}_frame_{frame_id}.jpg"
        return None, Path(filename).resolve()
    return _active_recorder.allocate_image_path(
        capture_id,
        frame_id,
        image_kind=image_kind,
    )


def register_result_image(
    image_id,
    image_path,
    capture_id,
    frame_id,
    summary,
):
    if _active_recorder is None or image_id is None:
        return
    _active_recorder.register_image(
        image_id,
        image_path,
        capture_id,
        frame_id,
        summary,
    )


def allocate_depth_diagnostic_paths(capture_id, frame_id, target_index):
    """返回当前运行目录中的深度诊断文件路径；未记录运行时返回空值。"""
    if _active_recorder is None:
        return None, None
    return _active_recorder.allocate_depth_diagnostic_paths(
        capture_id,
        frame_id,
        target_index,
    )


def finalize_run_recorder():
    global _active_recorder

    recorder = _active_recorder
    if recorder is None:
        return None

    try:
        return recorder.finalize()
    finally:
        _active_recorder = None

import numpy as np
import os

from .render_data import RenderPayload


class AttractorView3D:
    def __init__(self):
        from PySide6 import QtWidgets

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self._headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        if self._headless:
            label = QtWidgets.QLabel("3D OpenGL view disabled in offscreen mode")
            label.setStyleSheet("color: #888888; font-family: monospace;")
            layout.addWidget(label)
            return

        import pyqtgraph.opengl as gl

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((15, 34, 89))
        self.view.opts["distance"] = 120
        self.view.opts["elevation"] = 18
        self.view.opts["azimuth"] = -58
        layout.addWidget(self.view)

        self.grid = gl.GLGridItem()
        self.grid.setSize(120, 120)
        self.grid.setSpacing(10, 10)
        self.view.addItem(self.grid)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
            size=1.4,
            pxMode=True,
        )
        self.line = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(0.93, 0.93, 0.93, 0.38),
            width=1.0,
            mode="line_strip",
        )
        self.view.addItem(self.line)
        self.view.addItem(self.scatter)

    def set_payload(self, payload: RenderPayload) -> None:
        if self._headless:
            return

        empty = np.empty((0, 3), dtype=np.float32)
        if payload.show_points and payload.point_colors is not None:
            self.scatter.setData(
                pos=payload.positions,
                color=payload.point_colors,
                size=1.4,
                pxMode=True,
            )
            self.scatter.setVisible(True)
        else:
            self.scatter.setData(pos=empty)
            self.scatter.setVisible(False)

        if payload.show_lines:
            self.line.setData(
                pos=payload.positions,
                color=payload.line_color,
                width=1.0,
                mode="line_strip",
            )
            self.line.setVisible(True)
        else:
            self.line.setData(pos=empty)
            self.line.setVisible(False)


class ProjectionView:
    def __init__(self, title: str):
        from PySide6 import QtWidgets
        import pyqtgraph as pg

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QtWidgets.QLabel(title)
        label.setStyleSheet("color: #888888; font-family: monospace; font-size: 10px;")
        layout.addWidget(label)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMinimumHeight(110)
        layout.addWidget(self.plot)

        self.line_item = pg.PlotDataItem(
            pen=pg.mkPen((238, 238, 238, 140), width=1)
        )
        self.point_item = pg.PlotDataItem(
            pen=None,
            symbol="o",
            symbolSize=2,
            symbolBrush=pg.mkBrush(238, 238, 238, 190),
        )
        self.plot.addItem(self.line_item)
        self.plot.addItem(self.point_item)

    def set_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        show_points: bool,
        show_lines: bool,
    ) -> None:
        if show_lines:
            self.line_item.setData(x, y)
        else:
            self.line_item.clear()

        if show_points:
            self.point_item.setData(x, y)
        else:
            self.point_item.clear()

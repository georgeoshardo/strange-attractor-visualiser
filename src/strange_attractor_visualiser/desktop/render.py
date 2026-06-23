import numpy as np
import os

from .render_data import RenderPayload, ViewBounds


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

        self._fit_camera(payload.view_bounds)
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
                pos=payload.line_positions,
                color=payload.line_colors,
                width=1.0,
                mode="line_strip",
            )
            self.line.setVisible(True)
        else:
            self.line.setData(pos=empty)
            self.line.setVisible(False)

    def _fit_camera(self, bounds: ViewBounds) -> None:
        from pyqtgraph import Vector

        center = bounds.center
        self.view.setCameraPosition(
            pos=Vector(float(center[0]), float(center[1]), float(center[2])),
            distance=float(bounds.camera_distance),
        )
        self.grid.setSize(x=float(bounds.grid_size), y=float(bounds.grid_size))
        self.grid.setSpacing(
            x=float(bounds.grid_spacing),
            y=float(bounds.grid_spacing),
        )
        self.grid.resetTransform()
        self.grid.translate(
            float(center[0]),
            float(center[1]),
            float(bounds.minimum[2]),
        )


class ProjectionView:
    def __init__(self, title: str):
        from PySide6 import QtCore, QtGui, QtWidgets
        import pyqtgraph as pg

        class _ColoredLineItem(pg.GraphicsObject):
            def __init__(self):
                super().__init__()
                self.picture = None
                self._bounds = QtCore.QRectF()

            def setData(
                self,
                x: np.ndarray,
                y: np.ndarray,
                colors: np.ndarray,
            ) -> None:
                self.prepareGeometryChange()
                if len(x) < 2:
                    self.picture = None
                    self._bounds = QtCore.QRectF()
                    self.update()
                    return

                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                colors = np.asarray(colors, dtype=float)
                self._bounds = QtCore.QRectF(
                    float(np.min(x)),
                    float(np.min(y)),
                    float(np.max(x) - np.min(x)),
                    float(np.max(y) - np.min(y)),
                )

                picture = QtGui.QPicture()
                painter = QtGui.QPainter(picture)
                pen = QtGui.QPen()
                pen.setWidthF(1.0)
                for index in range(len(x) - 1):
                    r, g, b, a = colors[index]
                    pen.setColor(
                        QtGui.QColor.fromRgbF(
                            float(r),
                            float(g),
                            float(b),
                            float(a),
                        )
                    )
                    painter.setPen(pen)
                    painter.drawLine(
                        QtCore.QPointF(float(x[index]), float(y[index])),
                        QtCore.QPointF(float(x[index + 1]), float(y[index + 1])),
                    )
                painter.end()
                self.picture = picture
                self.update()

            def clear(self) -> None:
                self.prepareGeometryChange()
                self.picture = None
                self._bounds = QtCore.QRectF()
                self.update()

            def paint(self, painter, *_args) -> None:
                if self.picture is not None:
                    self.picture.play(painter)

            def boundingRect(self):
                return self._bounds

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
        self.colored_line_item = _ColoredLineItem()
        self.point_item = pg.PlotDataItem(
            pen=None,
            symbol="o",
            symbolSize=2,
            symbolBrush=pg.mkBrush(238, 238, 238, 190),
        )
        self.plot.addItem(self.line_item)
        self.plot.addItem(self.colored_line_item)
        self.plot.addItem(self.point_item)

    def set_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        show_points: bool,
        show_lines: bool,
        line_colors: np.ndarray | None = None,
        line_x: np.ndarray | None = None,
        line_y: np.ndarray | None = None,
    ) -> None:
        if show_lines:
            line_x = x if line_x is None else line_x
            line_y = y if line_y is None else line_y
            if line_colors is not None:
                self.line_item.clear()
                self.colored_line_item.setData(line_x, line_y, line_colors)
            else:
                self.colored_line_item.clear()
                self.line_item.setData(line_x, line_y)
        else:
            self.line_item.clear()
            self.colored_line_item.clear()

        if show_points:
            self.point_item.setData(x, y)
        else:
            self.point_item.clear()

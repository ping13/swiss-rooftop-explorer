from qgis.PyQt.QtWidgets import QMessageBox
from qgis.gui import QgsMapTool
from qgis.core import (
    QgsProject,
    QgsPointXY,
    QgsGeometry,
    QgsWkbTypes,
    QgsVectorLayer
)

class FractionAlongLineTool(QgsMapTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.layer = canvas.currentLayer()

    def canvasReleaseEvent(self, event):
        point = self.canvas.getCoordinateTransform().toMapCoordinates(event.pos())
        clicked_geom = QgsGeometry.fromPointXY(QgsPointXY(point))

        if self.layer is None or self.layer.geometryType() != QgsWkbTypes.LineGeometry:
            QMessageBox.information(None, "Info", "Please select a line layer.")
            return

        nearest = None
        nearest_dist = float('inf')

        for feature in self.layer.getFeatures():
            geom = feature.geometry()
            if not geom:
                continue

            dist = geom.distance(clicked_geom)
            if dist < nearest_dist:
                nearest = feature
                nearest_dist = dist

        if nearest is not None:
            line_geom = nearest.geometry()
            loc = line_geom.lineLocatePoint(clicked_geom)
            total_length = line_geom.length()
            if total_length > 0:
                fraction = loc / total_length
                QMessageBox.information(None, "Fraction", f"Fraction along line: {fraction:.3f}")
            else:
                QMessageBox.warning(None, "Warning", "Line has zero length.")
        else:
            QMessageBox.information(None, "Info", "No nearby line found.")

tool = FractionAlongLineTool(iface.mapCanvas())
iface.mapCanvas().setMapTool(tool)
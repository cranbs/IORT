from PyQt5 import QtCore, QtWidgets, QtGui
from IORT.rect import Rect
from enum import Enum
from typing import List
from IORT.widgets.polygon import Polygon, PromptPoint
from IORT.configs import STATUSMode, DRAWMode, CONTOURMode, CLICKMode
import time
from PIL import Image
import numpy as np
import cv2

class Scene(QtWidgets.QGraphicsScene):
    def __init__(self, mainwindow):
        super(Scene, self).__init__()
        self.image_item = None
        self.mask_item = None
        self.current_sam_rect: Rect = None
        self.mainwindow = mainwindow
        self.pixmap_item:QtWidgets.QGraphicsPixmapItem = None
        self.rects:List[Rect] = []
        self.mode:STATUSMode = STATUSMode.VIEW
        self.current_rect:Rect = None
        self.leftpressed = False
        self.image_data = None
        self.current_graph:Polygon = None
        self.draw_mode = DRAWMode.POLYGON
        self.pressed = False
        self.draw_interval = 0.15
        self.current_line = None
        self.selected_polygons_list = list()
        self.guide_line_x:QtWidgets.QGraphicsLineItem = None
        self.guide_line_y:QtWidgets.QGraphicsLineItem = None
        self.click_points = []                              # SAM point prompt
        self.click_points_mode = [] 
        self.prompt_points = []
        self.masks: np.ndarray = None
        self.mask_alpha = 0.5
        self.contour_mode = CONTOURMode.SAVE_EXTERNAL       # 默认SAM只保留外轮廓
        self.selected_polygons_list = list()
    
    def load_image(self, image_path: str):
        self.clear()
        if self.mainwindow.use_segment_anything:
            self.mainwindow.segany.reset_image()
        
        image = Image.open(image_path)
        image = image.convert('RGB')
        self.image_data = np.array(image)
        self.image_item = QtWidgets.QGraphicsPixmapItem()
        self.image_item.setZValue(0)
        self.addItem(self.image_item)
        self.mask_item = QtWidgets.QGraphicsPixmapItem()
        self.mask_item.setZValue(1)
        self.addItem(self.mask_item)

        self.image_item.setPixmap(QtGui.QPixmap(image_path))
        self.setSceneRect(self.image_item.boundingRect())
        self.change_mode_to_view()
    
    def unload_image(self):
        self.clear()
        self.setSceneRect(QtCore.QRectF())
        self.image_item = None
        self.mask_item = None
        self.current_graph = None
    
    def start_segment_anything(self):
        self.draw_mode = DRAWMode.SEGMENTANYTHING
        self.start_draw()
    
    def start_draw_polygon(self):
        self.draw_mode = DRAWMode.POLYGON
        self.start_draw()
    
    def change_click_to_positive(self):
        self.click = CLICKMode.POSITIVE

    def change_click_to_negative(self):
        self.click = CLICKMode.NEGATIVE
        
    def start_draw(self):
        if self.mode != STATUSMode.VIEW:
            return
        self.change_mode_to_create()
        if self.mode == STATUSMode.CREATE:
            self.current_graph = Polygon()
            self.addItem(self.current_graph)
    
    def finish_draw(self):
        if self.current_graph is None:
            return
        
        if self.draw_mode == DRAWMode.POLYGON:
            if len(self.current_graph.points) < 1:
                return
            # 移除鼠标移动点
            self.current_graph.removePoint(len(self.current_graph.points) - 1)

            # 单点，删除
            if len(self.current_graph.points) < 2:
                self.current_graph.delete()
                self.removeItem(self.current_graph)

                self.change_mode_to_view()
                self.mainwindow.set_labels_visible(True)

                return

            # 两点，默认矩形
            if len(self.current_graph.points) == 2:
                first_point = self.current_graph.points[0]
                last_point = self.current_graph.points[-1]
                self.current_graph.removePoint(len(self.current_graph.points) - 1)
                self.current_graph.addPoint(QtCore.QPointF(first_point.x(), last_point.y()))
                self.current_graph.addPoint(last_point)
                self.current_graph.addPoint(QtCore.QPointF(last_point.x(), first_point.y()))

            self.current_graph.set_drawed(QtGui.QColor('#00A0FF'))
             
            # 设置为最高图层
            self.current_graph.setZValue(len(self.mainwindow.polygons)+1)
            for vertex in self.current_graph.vertices:
                vertex.setZValue(len(self.mainwindow.polygons)+1)

            # 添加新polygon
            self.mainwindow.polygons.append(self.current_graph)
        elif self.draw_mode == DRAWMode.SEGMENTANYTHING or self.draw_mode == DRAWMode.SEGMENTANYTHING_BOX:
            if self.masks is not None:
                masks = self.masks
                masks = masks.astype('uint8') * 255
                h, w = masks.shape[-2:]
                masks = masks.reshape(h, w)

                if self.contour_mode == CONTOURMode.SAVE_ALL:
                    # 当保留所有轮廓时，检测所有轮廓，并建立二层等级关系
                    contours, hierarchy = cv2.findContours(masks, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
                else:
                    # 当只保留外轮廓或单个mask时，只检测外轮廓
                    contours, hierarchy = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

                if self.contour_mode == CONTOURMode.SAVE_MAX_ONLY:
                    largest_contour = max(contours, key=cv2.contourArea)    # 只保留面积最大的轮廓
                    contours = [largest_contour]

                for index, contour in enumerate(contours):

                    epsilon_factor = 0.001
                    epsilon = epsilon_factor * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)

                    if self.current_graph is None:
                        self.current_graph = Polygon()
                        self.addItem(self.current_graph)

                    if len(contour) < 3:
                        continue
                    for point in contour:
                        x, y = point[0]
                        x = max(0.1, x)
                        y = max(0.1, y)
                        self.current_graph.addPoint(QtCore.QPointF(x, y))

                    self.current_graph.set_drawed(QtGui.QColor('#00A0FF'))
                    # 添加新polygon
                    self.mainwindow.polygons.append(self.current_graph)

                    self.current_graph = None
                self.masks = None
            
        self.current_graph = None

        self.change_mode_to_view()
        self.update_mask()
    
    def change_mode_to_create(self):
        if self.image_item == None:
            return
        self.mode = STATUSMode.CREATE
        if self.image_item is not None:
            self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.mainwindow.actionPrevious_image.setEnabled(False)
        self.mainwindow.actionNext_image.setEnabled(False)
        self.mainwindow.actionDelete.setEnabled(False)
        self.mainwindow.actionSave.setEnabled(False)
        self.mainwindow.actionObjectRemoval.setEnabled(False)
        self.mainwindow.actionSegment_anything_point.setEnabled(False)
        self.mainwindow.actionPolygon.setEnabled(False)
        self.mainwindow.actionFinish.setEnabled(True)
        self.mainwindow.actionCancel.setEnabled(True)
        self.mainwindow.actionVisible.setEnabled(True)
        #---下面添加一个状态栏显示当前模式---
        
    def change_mode_to_view(self):
        self.mode = STATUSMode.VIEW
        self.mainwindow.actionPrevious_image.setEnabled(True)
        self.mainwindow.actionNext_image.setEnabled(True)
        self.mainwindow.actionSave.setEnabled(True)
        self.mainwindow.actionObjectRemoval.setEnabled(True)
        self.mainwindow.actionSegment_anything_point.setEnabled(True)
        self.mainwindow.actionPolygon.setEnabled(True)
    
    def cancel_draw(self):
        if self.mode == STATUSMode.CREATE:
            if self.current_graph is not None:
                self.current_graph.delete()  # 清除所有路径
                self.removeItem(self.current_graph)
                self.current_graph = None
        if self.mode == STATUSMode.EDIT:
            for item in self.selectedItems():
                item.setSelected(False)

        if self.current_sam_rect is not None:
            self.current_sam_rect.delete()
            self.removeItem(self.current_sam_rect)
            self.current_sam_rect = None
        
        self.change_mode_to_view()
        
        # mask清空
        self.click_points.clear()
        self.click_points_mode.clear()
        for prompt_point in self.prompt_points:
            try:
                self.removeItem(prompt_point)
            finally:
                del prompt_point
        self.prompt_points.clear()
        self.update_mask()
    
    def update_mask(self):
        if not self.mainwindow.use_segment_anything:
            return
        
        if self.image_data is None:
            return
        if not (self.image_data.ndim == 3 and self.image_data.shape[-1] == 3):
            return
        if len(self.click_points) > 0 and len(self.click_points_mode) > 0:
            masks = self.mainwindow.segany.predict_with_point_prompt(self.click_points, self.click_points_mode)
            self.masks = masks
            color = np.array([0, 0, 255])
            h, w = masks.shape[-2:]
            mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image = mask_image.astype("uint8")
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            mask_image = cv2.addWeighted(self.image_data, self.mask_alpha, mask_image, 1, 0)
        elif self.current_sam_rect is not None:
            point1 = self.current_sam_rect.points[0]
            point2 = self.current_sam_rect.points[1]
            box = np.array([min(point1.x(), point2.x()),
                            min(point1.y(), point2.y()),
                            max(point1.x(), point2.x()),
                            max(point1.y(), point2.y()),
                            ])
            masks = self.mainwindow.segany.predict_with_box_prompt(box)

            self.masks = masks
            color = np.array([0, 0, 255])
            h, w = masks.shape[-2:]
            mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image = mask_image.astype("uint8")
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            # 这里通过调整原始图像的权重self.mask_alpha，来调整mask的明显程度。
            mask_image = cv2.addWeighted(self.image_data, self.mask_alpha, mask_image, 1, 0)
        else:
            mask_image = np.zeros(self.image_data.shape, dtype=np.uint8)
            mask_image = cv2.addWeighted(self.image_data, 1, mask_image, 0, 0)
        mask_image = QtGui.QImage(mask_image[:], mask_image.shape[1], mask_image.shape[0], mask_image.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)
        mask_pixmap = QtGui.QPixmap(mask_image)
        if self.mask_item is not None:
            self.mask_item.setPixmap(mask_pixmap)
        
        # mask_image = np.zeros(self.image_data.shape, dtype=np.uint8)
        # mask_image = cv2.addWeighted(self.image_data, 1, mask_image, 0, 0)
        # mask_image = QtGui.QImage(mask_image[:], mask_image.shape[1], mask_image.shape[0], mask_image.shape[1] * 3,
        #                           QtGui.QImage.Format_RGB888)
        # mask_pixmap = QtGui.QPixmap(mask_image)
        # if self.mask_item is not None:
        #     self.mask_item.setPixmap(mask_pixmap)
    
    def mousePressEvent(self, event: 'QtWidgets.QGraphicsSceneMouseEvent'):
        pos = event.scenePos()
        if pos.x() < 0: pos.setX(0)
        if pos.x() > self.width() - 1: pos.setX(self.width() - 1)
        if pos.y() < 0: pos.setY(0)
        if pos.y() > self.height() - 1: pos.setY(self.height() - 1)
        if self.mode == STATUSMode.CREATE:
            # 拖动鼠标描点
            self.last_draw_time = time.time()
            self.pressed = True

            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                if self.draw_mode == DRAWMode.POLYGON:
                    # 移除随鼠标移动的点
                    self.current_graph.removePoint(len(self.current_graph.points) - 1)
                    # 添加当前点
                    self.current_graph.addPoint(pos)
                    # 添加随鼠标移动的点
                    self.current_graph.addPoint(pos)
                elif self.draw_mode == DRAWMode.SEGMENTANYTHING:
                    self.click_points.append([pos.x(), pos.y()])
                    self.click_points_mode.append(1)
                    prompt_point = PromptPoint(pos, 1)
                    prompt_point.setVisible(True)
                    self.prompt_points.append(prompt_point)
                    self.addItem(prompt_point)
                else:
                    raise ValueError('The draw mode named {} not supported.')
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                if self.draw_mode == DRAWMode.POLYGON:
                    pass
                elif self.draw_mode == DRAWMode.SEGMENTANYTHING:
                    self.click_points.append([pos.x(), pos.y()])
                    self.click_points_mode.append(0)
                    prompt_point = PromptPoint(pos, 0)
                    prompt_point.setVisible(True)
                    self.prompt_points.append(prompt_point)
                    self.addItem(prompt_point)
                else:
                    raise ValueError('The draw mode named {} not supported.')
            if self.draw_mode == DRAWMode.SEGMENTANYTHING:
                self.update_mask()

        # self.mainwindow.plugin_manager_dialog.trigger_on_mouse_press(pos)

        super(Scene, self).mousePressEvent(event)

    # 拖动鼠标描点
    def mouseReleaseEvent(self, event: 'QtWidgets.QGraphicsSceneMouseEvent'):
        self.pressed = False

        pos = event.scenePos()
        if pos.x() < 0: pos.setX(0)
        if pos.x() > self.width() - 1: pos.setX(self.width() - 1)
        if pos.y() < 0: pos.setY(0)
        if pos.y() > self.height() - 1: pos.setY(self.height() - 1)

        super(Scene, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: 'QtWidgets.QGraphicsSceneMouseEvent'):
        # 辅助线
        if self.guide_line_x is not None and self.guide_line_y is not None:
            if self.guide_line_x in self.items():
                self.removeItem(self.guide_line_x)

            if self.guide_line_y in self.items():
                self.removeItem(self.guide_line_y)

            self.guide_line_x = None
            self.guide_line_y = None

        pos = event.scenePos()
        if pos.x() < 0: pos.setX(0)
        if pos.x() > self.width() - 1: pos.setX(self.width() - 1)
        if pos.y() < 0: pos.setY(0)
        if pos.y() > self.height() - 1: pos.setY(self.height() - 1)
        # 限制在图片范围内

        if self.mode == STATUSMode.CREATE:
            if self.draw_mode == DRAWMode.POLYGON:
                # 随鼠标位置实时更新多边形
                self.current_graph.movePoint(len(self.current_graph.points) - 1, pos)
            if self.draw_mode == DRAWMode.SEGMENTANYTHING_BOX:
                if self.current_sam_rect is not None:
                    self.current_sam_rect.movePoint(len(self.current_sam_rect.points) - 1, pos)
                    self.update_mask()

        pen = QtGui.QPen()
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        # 辅助线
        if self.guide_line_x is None and self.width() > 0 and self.height() > 0:
            self.guide_line_x = QtWidgets.QGraphicsLineItem(QtCore.QLineF(pos.x(), 0, pos.x(), self.height()))
            self.guide_line_x.setPen(pen)
            self.guide_line_x.setZValue(1)
            self.addItem(self.guide_line_x)
        if self.guide_line_y is None and self.width() > 0 and self.height() > 0:
            self.guide_line_y = QtWidgets.QGraphicsLineItem(QtCore.QLineF(0, pos.y(), self.width(), pos.y()))
            self.guide_line_y.setPen(pen)
            self.guide_line_y.setZValue(1)
            self.addItem(self.guide_line_y)

        # 状态栏,显示当前坐标
        if self.image_data is not None:
            x, y = round(pos.x()), round(pos.y())
            self.mainwindow.labelCoord.setText('xy: ({:>4d},{:>4d})'.format(x, y))
            data = self.image_data[y][x]
            if self.image_data.ndim == 3:
                if len(data) == 3:
                    self.mainwindow.labelData.setText('rgb: [{:>3d},{:>3d},{:>3d}]'.format(data[0], data[1], data[2]))
                else:
                    self.mainwindow.labelData.setText('pix: [{}]'.format(data))

        # 拖动鼠标描点
        if self.pressed:  # 拖动鼠标
            current_time = time.time()
            if self.last_draw_time is not None and current_time - self.last_draw_time < self.draw_interval:
                return  # 时间小于给定值不画点
            self.last_draw_time = current_time

            if self.current_graph is not None:
                if self.draw_mode == DRAWMode.POLYGON:
                    # 移除随鼠标移动的点
                    self.current_graph.removePoint(len(self.current_graph.points) - 1)
                    # 添加当前点
                    self.current_graph.addPoint(pos)
                    # 添加随鼠标移动的点
                    self.current_graph.addPoint(pos)

            if self.current_line is not None:
                # 移除随鼠标移动的点
                self.current_line.removePoint(len(self.current_line.points) - 1)
                # 添加当前点
                self.current_line.addPoint(pos)
                # 添加随鼠标移动的点
                self.current_line.addPoint(pos)

        super(Scene, self).mouseMoveEvent(event)


class View(QtWidgets.QGraphicsView):
    def __init__(self, parent):
        super(View, self).__init__(parent)
        self.factor = 0.8
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)

    def zoom(self, factor, point=None):
        mouse_old = self.mapToScene(point) if point is not None else None
        pix_widget = self.transform().scale(factor, factor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()
        if pix_widget > 10 or pix_widget < 0.01:
            return
        self.scale(factor,factor)
        if point is not None:
            mouse_now = self.mapToScene(point)
            center_now = self.mapToScene(self.viewport().width() // 2, self.viewport().height() // 2)
            center_new = mouse_old - mouse_now + center_now
            self.centerOn(center_new)

    def zoom_fit(self):
        self.fitInView(0, 0, self.scene().width(), self.scene().height(),  QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self):
        self.zoom(1 / self.factor)

    def zoom_out(self):
        self.zoom(self.factor)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        angel= event.angleDelta()
        if angel.y() < 0:
            self.zoom(self.factor, event.pos())
        elif angel.y() > 0:
            self.zoom(1/self.factor, event.pos())
        else:
            pass

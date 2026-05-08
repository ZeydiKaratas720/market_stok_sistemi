import sys
import cv2
import os
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QTableWidget,
    QTableWidgetItem, QMessageBox, QRadioButton, QHBoxLayout
)
from PyQt5.QtCore import Qt

# ---------------- GUI ----------------
class StokApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Market Stok Sistemi")
        self.setGeometry(100, 100, 400, 550)

        self.urun_label = QLabel("Algılanan Ürün: -")
        self.urun_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        # İşlem Türü Seçme
        self.radio_group_layout = QHBoxLayout()
        self.radio_ekle = QRadioButton("Stok Girişi (+)")
        self.radio_cikar = QRadioButton("Satış Yap (-)")
        self.radio_ekle.setChecked(True) 
        self.radio_group_layout.addWidget(self.radio_ekle)
        self.radio_group_layout.addWidget(self.radio_cikar)
        # -------------------------------

        self.input = QLineEdit()
        self.input.setPlaceholderText("Miktar giriniz")

        self.button = QPushButton("İşlemi Onayla")
        self.button.clicked.connect(self.stok_ekle)
        self.button.setStyleSheet("background-color: #4CAF50; color: white; height: 30px;")

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Ürün", "Stok"])
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout()
        layout.addWidget(self.urun_label)
        layout.addLayout(self.radio_group_layout) 
        layout.addWidget(self.input)
        layout.addWidget(self.button)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.current_product = None

    def set_product(self, urun):
        self.current_product = urun
        self.urun_label.setText(f"Algılanan Ürün: {urun}")

    def stok_ekle(self):
        if not self.current_product:
            QMessageBox.warning(self, "Hata", "Önce bir ürün algılanmalı!")
            return

        miktar_str = self.input.text()
        if not miktar_str.isdigit():
            QMessageBox.warning(self, "Hata", "Lütfen geçerli bir sayı giriniz!")
            return

        miktar = int(miktar_str)
        
        
        if self.radio_cikar.isChecked():
            miktar = -miktar

        # Tabloda mevcut ürünü ara
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == self.current_product:
                mevcut_stok = int(self.table.item(row, 1).text())
                yeni_stok = mevcut_stok + miktar
                
                
                if yeni_stok < 0:
                    QMessageBox.critical(self, "Hata", f"Yetersiz stok! Mevcut: {mevcut_stok}")
                    return
                
                self.table.setItem(row, 1, QTableWidgetItem(str(yeni_stok)))
                self.input.clear()
                return

        
        if miktar < 0:
            QMessageBox.warning(self, "Hata", "Stokta olmayan bir ürünün satışı yapılamaz!")
            return

        
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(self.current_product))
        self.table.setItem(row, 1, QTableWidgetItem(str(miktar)))
        self.input.clear()

# ---------------- MAIN ----------------
def main():
    app = QApplication(sys.argv)
    window = StokApp()
    window.show()
    
    app.processEvents()

    
    
    model_path = "/Users/dilarakaratas/Desktop/market_stok_takibi/runs/detect/train4/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı! Yol: {model_path}")
        return
    
    print("Model yükleniyor...")
    model = YOLO(model_path)

    # 2. Kamera Kontrolü
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("HATA: Kamera açılamadı! Lütfen izinleri kontrol edin.")
        return

    frame_count = 0
    CONF_THRESHOLD = 0.6
    last_label = None

    print("Sistem hazır. Kamera açılıyor... (Çıkış için 'q' tuşuna basın)")

    while True:
        app.processEvents()

        ret, frame = cap.read()
        if not ret:
            print("Kameradan görüntü alınamıyor.")
            break

        frame_count += 1

        if frame_count % 5 == 0:
            results = model(frame, verbose=False)
            
            best_box = None
            best_conf = 0

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box

            if best_box and best_conf > CONF_THRESHOLD:
                cls_id = int(best_box.cls[0])
                label = model.names[cls_id]

                if label != last_label:
                    window.set_product(label)
                    last_label = label

                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {best_conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Kamera Takip (Cikmak icin Q)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if cv2.getWindowProperty("Kamera Takip (Cikmak icin Q)", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Uygulama kapatıldı.")

if __name__ == "__main__":
    main()
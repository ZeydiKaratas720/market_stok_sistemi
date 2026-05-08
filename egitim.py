from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 'yolov8n.pt' yerine en son kaydedilen 'last.pt' dosyasını yüklüyoruz
    model = YOLO('runs/detect/train4/weights/last.pt') 

    # 2. 'resume=True' diyerek her şeyi kaldığı yerden (27. epoch) başlatıyoruz
    # Diğer ayarları (epochs, imgsz vb.) tekrar yazmana gerek yok, 
    # çünkü YOLO bunları 'last.pt' içinden otomatik okur.
    model.train(resume=True)
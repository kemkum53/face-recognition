# face_recognition ve opencv kütüphanelerini import ederek başlıyoruz
import face_recognition
import cv2


# opencv metodu olan VideoCapture ile webcam'den görüntü almayı başlatıyoruz // 0 default webcam 
video_capture = cv2.VideoCapture(0)
# Yukarıdaki "mennan sevim" resmini yüklüyoruz ve encoding bilgisini alıyoruz
mennan_image = face_recognition.load_image_file("kemal.jpg")
mennan_face_encoding = face_recognition.face_encodings(mennan_image)[0]

# Yukarıdaki "miray sevim" resmini yüklüyoruz ve encoding bilgisini alıyoruz
miray_image = face_recognition.load_image_file("mali.jpg")
miray_face_encoding = face_recognition.face_encodings(miray_image)[0]


# Encoding ve açıklama kısmını burada tanımlıyoruz, birden fazla tanımlayabiliriz
known_face_encodings = [
    mennan_face_encoding,
    miray_face_encoding,
]
known_face_names = [
    "Kemal",
    "Mustafa Ali"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
__nowPlaying = False

while True:
    # Videodan anlık bir kare yakalıyoruz
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    
    # Aldığımız kareyi 1/4 oranında küçültüyoruz ve bu daha hızlı sonuç vermeyi sağlıyor
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # BGR(opencv) türündeki resmi RGB(face_recognition) formatına çeviriyoruz
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Uyumlu tüm yüzlerin lokasyonlarını bulan kodlar
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Eşleşen yüzleri topla
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Bilinmeyen"

            
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)
            
            
    process_this_frame = not process_this_frame
    print(len(face_names))

    # Sonuçları göster
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Yüzü çerçeve içerisine al
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # "Mennan Sevim" etiketini oluştur
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Oluşan çerçeveyi ekrana yansıt
    cv2.imshow('Video', frame)

    # Çıkış için 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat
video_capture.release()
cv2.destroyAllWindows()
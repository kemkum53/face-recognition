###                               ###
#   201713709012 - Kemal Kondakçı   #
#  201713709015 - Mustafa Ali Eren  #
###                               ###
import face_recognition
import cv2


# opencv ile kameraya ulaşıyoruz.
video_capture = cv2.VideoCapture(0)
# Yüz tanıma sisteminde bulunmasını istediğimiz kişilerin fotoğraflarını sisteme yüklüyoruz, istediğimiz sayıda yükleme yapabiliriz.
kemal_image = face_recognition.load_image_file("kemalkondakci.jpg")
kemal_face_encoding = face_recognition.face_encodings(kemal_image)[0]

mali_image = face_recognition.load_image_file("mustafalieren.jpg")
mali_face_encoding = face_recognition.face_encodings(mali_image)[0]

huseyingunes_image = face_recognition.load_image_file("huseyingunes.jpg")
huseyingunes_face_encoding = face_recognition.face_encodings(huseyingunes_image)[0]


# Encodingleri burada kişilerle eşliyoruz.
known_face_encodings = [
    kemal_face_encoding,
    mali_face_encoding,
    huseyingunes_face_encoding
]
known_face_names = [
    "Kemal",
    "Mustafa Ali",
    "Huseyin Gunes"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
__nowPlaying = False

while True:
    # Videodan anlık kareler yakalıyoruz
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    # Dışardan resim yükleme istersek yukarıdaki 2 satırı yorum satırı yapıp bu 2 satırı açın
    # daha sonra 'deneme2.jpeg' yazan yere resimin konumunu yazın
    # frame = cv2.imread('deneme2.jpeg')
    # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Yakaladığımız kareleri 0.25'e küçültüyoruz ki programımız hızlı bir şekilde çalışabilisin.
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # BGR(opencv) türündeki resmi RGB(face_recognition) formatına çeviriyoruz
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Kamerada gözüken tüm yüzler
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Öğrettiğimiz yüzleri toplayan, diğerlerine "Bilinmeyen" adı veren kod bloğu
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Bilinmeyen"

            
            # known_face_encodings'de bir eşi olan yüzlerin burada isimlerini eşliyoruz
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)
            
            
    process_this_frame = not process_this_frame
    print(len(face_names))

    # Sonuçlar
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Yüzün etrafına bir çerçeve koyup,
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # İsimleri  yazdırıyoruz.
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Az önce yapttığımız işlemi ekranda gösteriyoruz.
    cv2.imshow('Video', frame)

    # Çıkış için 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat
video_capture.release()
cv2.destroyAllWindows()
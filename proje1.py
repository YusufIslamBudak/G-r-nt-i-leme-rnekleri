import cv2
from matplotlib import pyplot as plt
renkli_resim = cv2.imread("mustakil_ev.jpg")
gri_resim = cv2.cvtColor(renkli_resim, cv2.COLOR_BGR2GRAY)
cv2.rectangle(gri_resim, (0, 0), (50, 50), (0, 0, 0), -1)
cv2.rectangle(gri_resim, (gri_resim.shape[1]-50, 0), (gri_resim.shape[1], 50), (0, 0, 0), -1)
cv2.rectangle(gri_resim, (0, gri_resim.shape[0]-50), (50, gri_resim.shape[0]), (0, 0, 0), -1)
cv2.rectangle(gri_resim, (gri_resim.shape[1]-50, gri_resim.shape[0]-50), (gri_resim.shape[1], gri_resim.shape[0]), (0, 0, 0), -1)
cv2.imwrite("guncellenmis_ev.jpg", gri_resim)
ucak = cv2.imread("beyaz_ucak.jpg")
ucak = cv2.resize(ucak, (100, 100))
x_offset, y_offset = 50, 10 
renkli_resim[y_offset:y_offset+ucak.shape[0], x_offset:x_offset+ucak.shape[1]] = ucak
cv2.imwrite("ucakli_ev.jpg", renkli_resim)
kesit = renkli_resim[200:300, 100:200]  
target_y, target_x = 0, 400  
renkli_resim[target_y:target_y+100, target_x:target_x+100] = kesit
cv2.imwrite("eklenmis_ev.jpg", renkli_resim)
(h, w) = renkli_resim.shape[:2]
merkez = (w // 2, h // 2)
matris_45 = cv2.getRotationMatrix2D(merkez, 45, 1.0)
dondurulmus_45 = cv2.warpAffine(renkli_resim, matris_45, (w, h))
matris_60 = cv2.getRotationMatrix2D(merkez, -60, 1.0)
dondurulmus_60 = cv2.warpAffine(renkli_resim, matris_60, (w, h))
cv2.imwrite("45_derece.jpg", dondurulmus_45)
cv2.imwrite("60_derece.jpg", dondurulmus_60)
ucak_template = renkli_resim[y_offset:y_offset+ucak.shape[0], x_offset:x_offset+ucak.shape[1]] 
result = cv2.matchTemplate(renkli_resim, ucak_template, cv2.TM_CCOEFF_NORMED)
(_, max_val, _, max_loc) = cv2.minMaxLoc(result)
(w, h) = ucak_template.shape[1], ucak_template.shape[0]
cv2.rectangle(renkli_resim, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)
cv2.imwrite("son_soru.jpg", renkli_resim)

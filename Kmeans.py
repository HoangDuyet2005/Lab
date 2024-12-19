import numpy as np
import pandas as pd
import time
class KMeans:
    def __init__(self,so_cum,so_lan_lap_max=1000,sai_so=1e-3):
        self.so_cum= so_cum#số cụm
        self.so_lan_lap_max=so_lan_lap_max#số lần lặp max
        self.tam_cum=None#toạ độ tâm cụm
        self.nhan= None#nhãn cụm
        self.sai_so=sai_so#điều kiện dừng (sai số nhỏ)
    def khoi_tao_tam_cum(self,du_lieu):
        np.random.seed(0)
        chi_so_ngau_nhien=np.random.choice(du_lieu.shape[0],self.so_cum ,replace=False)
        return du_lieu[chi_so_ngau_nhien]
    def gan_nhan(self,du_lieu):#gán nhãn
        nhan=[]
        for i in du_lieu:#kc ơ cờ lít giữa các điểm và tâm cụm
            khoang_cach=np.linalg.norm(i-self.tam_cum, axis=1)
            nhan.append(np.argmin(khoang_cach))#chọn tâm cụm
        return np.array(nhan)
    def cap_nhat_tam_cum(self,du_lieu): #cập nhật kiểm tra lại tâm cụm
        tam_cum_moi=np.zeros((self.so_cum,du_lieu.shape[1]))
        for i in range(self.so_cum):
            diem_trong_cum=du_lieu[self.nhan==i]
            if len(diem_trong_cum)>0:
                tam_cum_moi[i]=diem_trong_cum.mean(axis=0)#tính tbc các điểm 1 cụm
        return tam_cum_moi
    def KM(self,du_lieu):
        self.tam_cum=self.khoi_tao_tam_cum(du_lieu)#khởi tạo tâm cụm ngẫu nhiên từ dl
        #self.dem=0
        for i in range(self.so_lan_lap_max):
            self.nhan= self.gan_nhan(du_lieu)#gán nhãn cho mỗi điểm dl
            tam_cum_moi=self.cap_nhat_tam_cum(du_lieu)#update tâm cụm
            #self.dem+=1
            if np.linalg.norm(self.tam_cum- tam_cum_moi)<self.sai_so:#so sánh(điều kiện dừng)
                break
            self.tam_cum=tam_cum_moi
        self.dem=i+1
    def lay_tam_cum(self):
        return self.tam_cum#lấy tâm cụm hiện tại
    def lay_nhan(self):
        return self.nhan#lấy nhãn hiện tại
    def lay_dem(self):
        return self.dem#lấy số lần lặp
def doc_flie(tendulieu):
    file=tendulieu+'.csv'
    du_lieu=pd.read_csv(file)
    thuoctinh=du_lieu.iloc[:,:-1].values #bỏ cột nhãn
    lay_nhan=np.array(du_lieu.iloc[:,-1].values)#lấy nhãn
    socum=len(np.unique(lay_nhan))#lấy số cụm
    return socum,thuoctinh
if __name__=="__main__":
    socum,thuoctinh=doc_flie('Iris')
    kmeans=KMeans(so_cum=socum)
    tg_bat_dau=time.time() #tg bắt đầu
    kmeans.KM(thuoctinh)
    tg_ket_thuc=time.time() #tg kết thúc
    print("Toạ độ tâm cụm")
    print(kmeans.lay_tam_cum())
    print("\nPhân cụm")
    print(kmeans.lay_nhan())
    print(f'Thời gian chạy: {tg_ket_thuc-tg_bat_dau:.5f}s')
    print(f'Số lần lặp: {kmeans.lay_dem()}')

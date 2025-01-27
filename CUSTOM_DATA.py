class Custom_Dataset(torch.utils.data.Dataset): ##커스텀 데이타셋 불러오는 폼


    def __init__(self, X, Y, transform=None):  #transform : augmentation x:입력 y:정답

        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):

        return self.X.shape[0] #개체행열 중 개

    def __getitem__(self,idx):   #인덱스
        x = self.X[idx]
        self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        y = self.Y[idx]

        return x,y 






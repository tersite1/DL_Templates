custom_DS = Custom_Dataset(x_data,y_data,transform=transform)


train_DS, val_DS, test_DS = torch.utils.data.random_split(custom_DS, [10,6,6])

'''

NoT = int(len(custom_DS)*0.8)
NoV = int(len(custom_DS)*0.1)
NoTes = len(custom_DS) - NoT - NoV

train_DS, val_DS, test_DS = torch.utils.data.random_split(custom_DS,[NoT,NoV,NoTes])

'''

train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=True)




for x_batch, y_batch in train_DL:
    print(f"x_batch = {x_batch.reshape(-1)}, \n" 
          f"y_batch = {y_batch.reshape(-1)}")  
print()
for x_batch, y_batch in val_DL:
    print(f"x_batch = {x_batch.reshape(-1)}, \n" 
          f"y_batch = {y_batch.reshape(-1)}")  
print()
for x_batch, y_batch in test_DL:
    print(f"x_batch = {x_batch.reshape(-1)}, \n" 
          f"y_batch = {y_batch.reshape(-1)}")  


print([p for p in model.parameters() if p.requires_grad])

for p in model.parameters():
    p.requires_grad=False




model.fc_out = nn.Linear(4,10) #모델 마지막 레이어를 새로 init, requires grad = true 인 상태 

optimizer = optim.Adam(params, lr=0.1) #true 인 grad 만 학습


print(list(model.modules())

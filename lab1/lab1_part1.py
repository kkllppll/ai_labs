import torch
import torch.nn as nn

# вхідні дані "AND", "OR", XOR"
truth_table = [
    # x1, x2
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]

# очікувані вихідні дані для логічної операції "AND", "OR", XOR"
expected_output_and = [0, 0, 0, 1]
expected_output_or = [0, 1, 1, 1]
expected_output_xor = [0, 1, 1, 0]

# вхідні дані для операції НІ
input_not = torch.tensor([[0], [1]], dtype=torch.float32)
# очікувані вихідні дані для операції НІ
expected_output_not = torch.tensor([[1], [0]], dtype=torch.float32)

# перетворення вхідних та вихідних даних у тензори PyTorch
inputs = torch.tensor(truth_table, dtype=torch.float32)
targets_and = torch.tensor(expected_output_and, dtype=torch.float32).view(-1, 1)
targets_or = torch.tensor(expected_output_or, dtype=torch.float32).view(-1, 1)
targets_xor = torch.tensor(expected_output_xor, dtype=torch.float32).view(-1, 1)

# клас для нейронної мережі операції НІ (NOT)
class NotNeuralNetwork(nn.Module):
    def __init__(self):
        super(NotNeuralNetwork, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        out = torch.sigmoid(self.fc(x))
        return out


# ініціалізація нейронної мережі операції НІ (NOT)
model_not = NotNeuralNetwork()

# визначення функції втра для навчання для операції НІ
criterion_not = nn.MSELoss()
# оптимізатор для моделі для оновлення вагових коефіцієнтів НІ
optimizer_not = torch.optim.SGD(model_not.parameters(), lr=0.1)

# клас для нейронної мережі
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        out = torch.sigmoid(self.fc(x))
        return out
    
# клас для багатошарової нейронної мережі
class XORNeuralNetwork(nn.Module):
    def __init__(self):
        super(XORNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 8)  
        self.fc2 = nn.Linear(8, 1)  

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # активація вхідн шару
        x = torch.sigmoid(self.fc2(x))  # активація вихідного шару
        return x
    
# ініціалізація нейронної мережі
model_and = NeuralNetwork()
model_or = NeuralNetwork()
model_xor = XORNeuralNetwork()


# визначення функції втрат для навчання моделей
criterion = nn.MSELoss()
# оптимізатор для моделі для оновлення вагових коефіцієнтів 
optimizer_and = torch.optim.SGD(model_and.parameters(), lr=0.1)
optimizer_or = torch.optim.SGD(model_or.parameters(), lr=0.1)
optimizer = torch.optim.SGD(model_xor.parameters(), lr=0.1)

# навчання нейронної мережі
for epoch in range(1000):
    #  обнуляють градієнти (похідні функції втрат по параметрам моделі) 
    # перед початком обчислень у кожній ітерації
    optimizer_and.zero_grad()
    optimizer_or.zero_grad()
    
    # передача вхідних даних
    out_and = model_and(inputs)
    out_or = model_or(inputs)
    
    # значення втрат
    loss_and = criterion(out_and, targets_and)
    loss_or = criterion(out_or, targets_or)
    

    # зворотнє поширення помилки (backpropagation)
    loss_and.backward()
    loss_or.backward()
    

    # оновлення параметрів моделей
    optimizer_and.step()
    optimizer_or.step()
    

# навчання моделі операції НІ (NOT)
for epoch in range(1000):
    optimizer_not.zero_grad()

    out_not = model_not(input_not)

    loss_not = criterion_not(out_not, expected_output_not)

    loss_not.backward()

    optimizer_not.step()


for epoch in range(10000):  
    optimizer.zero_grad()
    outputs = model_xor(inputs)
    loss = criterion(outputs, targets_xor)
    loss.backward()
    optimizer.step()

# тестування нейронних мереж
with torch.no_grad():
    test_out_and = model_and(inputs)
    test_out_or = model_or(inputs)
    test_out_xor = model_xor(inputs)
    test_out_not = model_not(input_not)
    test_out_xor = model_xor(inputs)
    
    predicted_outputs_and = torch.round(test_out_and)
    predicted_outputs_or = torch.round(test_out_or)
    predicted_outputs_not = torch.round(test_out_not)
    predicted_outputs_xor = torch.round(test_out_xor)

# виведення результатів
logic_operations = {
    "AND": predicted_outputs_and,
    "OR": predicted_outputs_or,
    "NOT": predicted_outputs_not,
    "XOR": predicted_outputs_xor
}
for operation, outputs in logic_operations.items():
    print(f"\nPredicted Outputs for {operation}:")
    for i, output in enumerate(outputs):
        if operation == "NOT":
            print(int(input_not[i].item()), "->", int(output.item()))
        else:
            print(truth_table[i], "->", int(output.item()))

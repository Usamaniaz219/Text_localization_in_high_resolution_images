data = [
    [1057, 159, 1200, 159, 1200, 197, 1057, 197],
    [1213, 159, 1367, 159, 1367, 195, 1213, 195],
    [2338, 164, 2394, 164, 2394, 188, 2338, 188],
    [2445, 125, 2493, 125, 2493, 145, 2445, 145],
    [2505, 127, 2537, 127, 2537, 145, 2505, 145],
    [3048, 126, 3102, 126, 3102, 150, 3048, 150],
    [3174, 150, 3200, 150, 3200, 174, 3174, 174],
    [3200, 150, 3228, 150, 3228, 174, 3200, 174],
    [3287, 131, 3400, 131, 3400, 167, 3287, 167],
    [3400, 131, 3577, 131, 3577, 167, 3400, 167],
    [3764, 152, 3800, 152, 3800, 178, 3764, 178],
    [3707, 189, 3781, 189, 3781, 200, 3707, 200],
    [3800, 150, 3842, 150, 3842, 180, 3800, 180],
    [3858, 164, 3962, 164, 3962, 196, 3858, 196]
]

for sublist in data:
    csv_values = ','.join(str(value) for value in sublist)
    print(csv_values)



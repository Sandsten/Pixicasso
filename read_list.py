import numpy as np
style_name_list = ["kandinsky", "shipwreck", "the_scream","seated-nude", "starry-night", "woman-with-hat-matisse"]
loss_types = ['total_loss', 'content_loss', 'style_loss', 'cross_loss']
for loss in loss_types:
  values = []
  for name in style_name_list:
    data = np.load("results_1_faster_decay/data_list_"+name+".npz")['arr_0'].item()
    values.append((name, data[loss][-1]))
    # values.append((name, np.sum(data[loss])/1000))
  values.sort(key=lambda tup: tup[1])
  print(loss)
  print(*values, sep="\n")

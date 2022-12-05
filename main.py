import dataa
import ad

d = dataa.WineDataset("C:/Users/lul-0/PycharmProjects/GradientBoosting/wine-quality-white-and-red 2.csv")
dannn = d()
x_train, t_train, x_test, t_test = dannn['train_input'], dannn['train_target'], dannn['test_input'], dannn['test_target']
aadd = ad.AD(20, 0.05)
aadd.gradBoost(x_train,t_train)
print(aadd.MSE(x_train,t_train))
print(aadd.MSE(x_test,t_test))
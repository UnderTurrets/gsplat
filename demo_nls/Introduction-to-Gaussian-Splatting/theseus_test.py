'''
Created by Han Xu
email:736946693@qq.com
'''
import torch

torch.manual_seed(0)

def generate_data(num_points=100, a=3, b=0.5, noise_factor=0.01):
    # Generate data: 100 points sampled from the quadratic curve listed above
    data_x = torch.rand((1, num_points))
    noise = torch.randn((1, num_points)) * noise_factor
    data_y = a * data_x.square() + b + noise
    return data_x, data_y

data_x, data_y = generate_data()

# Plot the data
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(data_x, data_y);
ax.set_xlabel('x');
ax.set_ylabel('y');


import theseus as th

# data is of type Variable
x = th.Variable(tensor=data_x, name="x")
y = th.Variable(tensor=data_y, name="y")

# optimization variables are of type Vector with 1 degree of freedom (dof)
a = th.Vector(1, name="a")
b = th.Vector(1, name="b")




def quad_error_fn(optim_vars, aux_vars):
    a, b = optim_vars
    x, y = aux_vars
    est = a.tensor * x.tensor.square() + b.tensor
    err = y.tensor - est
    return err

optim_vars = a, b
aux_vars = x, y

print(quad_error_fn(optim_vars,aux_vars))

cost_function = th.AutoDiffCostFunction(
    optim_vars, quad_error_fn, 100, aux_vars=aux_vars, name="quadratic_cost_fn"
)
objective = th.Objective()
objective.add(cost_function)
# optimizer = th.GaussNewton(
#     objective,
#     max_iterations=15,
#     step_size=0.5,
# )
optimizer = th.LevenbergMarquardt(
    objective=objective
)
theseus_optim = th.TheseusLayer(optimizer)

theseus_inputs = {
    "x": data_x,
    "y": data_y,

    # initial values
    "a": torch.rand(size=(1, 1)),
    "b": torch.rand(size=(1, 1))
}
with torch.no_grad():
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose": True})
print("Best solution:", info.best_solution)

# Plot the optimized function
fig, ax = plt.subplots()
ax.scatter(data_x, data_y);

a = info.best_solution['a'].squeeze()
b = info.best_solution['b'].squeeze()
x = torch.linspace(0., 1., steps=100)
y = a * x * x + b
ax.plot(x, y, color='k', lw=4, linestyle='--',
        label='Optimized quadratic')
ax.legend()

ax.set_xlabel('x');
ax.set_ylabel('y');


def cauchy_fn(x):
    return torch.sqrt(0.5 * torch.log(1 + x ** 2))

def cauchy_loss_quad_error_fn(optim_vars, aux_vars):
    err = quad_error_fn(optim_vars, aux_vars)
    return cauchy_fn(err)

wt_cost_function = th.AutoDiffCostFunction(
    optim_vars, cauchy_loss_quad_error_fn, 100, aux_vars=aux_vars, name="cauchy_quad_cost_fn"
)


objective = th.Objective()
objective.add(wt_cost_function)
optimizer = th.GaussNewton(
    objective,
    max_iterations=20,
    step_size=0.3,
)
theseus_optim = th.TheseusLayer(optimizer)
theseus_inputs = {
"x": data_x,
"y": data_y,
"a": 2 * torch.ones((1, 1)),
"b": torch.ones((1, 1))
}

# We suppress warnings in this optimization call, because we observed that with this data, Cauchy
# loss often results in singular systems with numerical computations as it approaches optimality.
# Please note: getting a singular system during the forward optimization will throw
# an error if torch's gradient tracking is enabled.
import warnings
warnings.simplefilter("ignore")

with torch.no_grad():
    _, info = theseus_optim.forward(
        theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose":True})
print("Best solution:", info.best_solution)

# Plot the optimized function
fig, ax = plt.subplots()
ax.scatter(data_x, data_y);

a = info.best_solution['a'].squeeze()
b = info.best_solution['b'].squeeze()
x = torch.linspace(0., 1., steps=100)
y = a*x*x + b
ax.plot(x, y, color='k', lw=4, linestyle='--',
        label='Optimized quadratic')
ax.legend()

ax.set_xlabel('x');
ax.set_ylabel('y');
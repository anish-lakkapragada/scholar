defmodule Scholar.Linear.RidgeRegression do
  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :bias], keep: [:alpha]}
  defstruct [:coefficients, :bias, :alpha]

  defn fit(x, y, opts \\ []) do
    # X is shape (N, b) and y is shape (N, 1)

    opts = keyword!(opts, [:alpha, 1])

    k = elem(Nx.shape(x), 1)

    %__MODULE__{coefficients: Nx.random_normal({k}), bias: Nx.random_normal({1}), alpha: opts[:alpha]}

    inverse_matrix = fit_closed_form(x, y, opts)
  end

  defnp fit_closed_form(x, y, opts) do
    alpha = opts[:alpha]
    x_transpose = x |> Nx.transpose()
    n = elem(Nx.shape(x), 0)
    ridge_adjustment = alpha * Nx.eye({n, n})
    inverse_matrix_part1 = x_transpose |> Nx.dot(x) |> Nx.add(ridge_adjustment) |> Nx.invert()
    inverse_matrix_part1
  end
end

x = Nx.random_normal(10, 1)
y = Nx.multiply(x, 2)

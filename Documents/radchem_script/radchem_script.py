import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf

""""This is version 1.7 of the curve fitting script
 for the RadChem lab. It includes a base class for
handling data processing, plotting, and residual
analysis, as well as specific classes for
exponential and linear fitting. The script
also incorporates error handling and confidence
interval calculations to provide a comprehensive
analysis of the fit quality."""
# Added NDF to compare everyone's data. 

class BaseFitter:
    """Parent class to handle plotting and residual calculations."""
    def __init__(self, x, y, xaxis_label, yaxis_label, background_subtract=None, dead_time = 0, is_rate = False, t_gross = 1.0, t_bg = 5.0):
        self.x = np.array(x)
        self.y = np.array(y)
        self.xaxis_label = xaxis_label
        self.yaxis_label = yaxis_label
        self.background_subtract = background_subtract
        self.dead_time = dead_time
        self.use_log = False
        self.is_rate = is_rate
        self.t_gross = t_gross
        self.t_bg = t_bg
        
        # Parameters to be filled by the child classes
        self.popt = None
        self.perr = None
        self.pcov = None

    def _process_data(self, linear_log=False):
        y_proc = self.y.copy().astype(float)
        
        # 1. Background (Must be first, on raw counts)
        if self.background_subtract is not None:
            y_proc = y_proc - np.mean(self.background_subtract)

        # 2. Dead-time
        if self.dead_time > 0:
            mask = y_proc > 5000
            y_proc[mask] = y_proc[mask] / (1 - y_proc[mask] * self.dead_time)
        
        # 3. Handle Transformation and Weights
        if self.is_rate:
            sigma_raw = np.sqrt(self.y / self.t_gross + np.mean(self.background_subtract)/self.t_bg)
        else:
            sigma_raw = np.sqrt(self.y + np.mean(self.background_subtract))
        if linear_log:
            epsilon = 1e-9
            y_proc1 = np.maximum(y_proc, epsilon)
            y_final = np.log(y_proc1)
            sigma = sigma_raw / y_proc1 # Correct log weighting
        else:
            y_final = y_proc
            sigma = sigma_raw # Raw Poisson weighting
        return y_final, sigma

    def calculate_confidence_interval(self, x_range, n_sigma=1.96):
        """Calculates the 95% (1.96 sigma) confidence band using the Delta Method."""
        if self.popt is None or self.pcov is None:
            return None, None
        
        y_fit = self.model(x_range, *self.popt)
        sigmas = []
        
        # We calculate the gradient (Jacobian) for each x point
        for x in x_range:
            grad = []
            eps = 1e-6 # Small step for numerical differentiation
            for i in range(len(self.popt)):
                p_plus = self.popt.copy()
                p_plus[i] += eps
                # f'(p) ≈ (f(p+eps) - f(p)) / eps
                derivative = (self.model(x, *p_plus) - self.model(x, *self.popt)) / eps
                grad.append(derivative)
            
            grad = np.array(grad)
            # Variance propagation: sigma_f^2 = g^T * Cov * g
            var_f = grad.T @ self.pcov @ grad
            sigmas.append(np.sqrt(var_f))
        
        sigmas = np.array(sigmas)
        return y_fit - n_sigma * sigmas, y_fit + n_sigma * sigmas

    def calculate_residuals(self):
        if self.popt is None: return None
        # Use self.use_log flag to get the matching data scale
        y_proc, _ = self._process_data(linear_log=getattr(self, 'use_log', False))
        return y_proc - self.model(self.x, *self.popt)

    def calculate_reduced_chi_sq(self):
        if self.popt is None:
            return 0
        
        # Get processed data and the matching sigma for this fit mode
        y_proc, sigma = self._process_data(linear_log=getattr(self, 'use_log', False))
        
        residuals = y_proc - self.model(self.x, *self.popt)
        
        # Chi-sq is the sum of (residual / uncertainty)^2
        chi_sq = np.sum((residuals / sigma)**2)
        
        # Degrees of Freedom = N - n_parameters
        dof = len(self.x) - len(self.popt)
        
        return chi_sq / dof if dof > 0 else 0

    def plot_fit(self, title=None, line_color='red'):
        # Get the processed data and the calculated sigma (error bars)
        y_proc, sigma = self._process_data(linear_log=self.use_log)
        plt.figure(figsize=(8, 5))
        plt.errorbar(self.x, y_proc, yerr=sigma, fmt='o', 
                     label='Data Points', color='blue', 
                     ecolor='black', elinewidth=1, capsize=2, alpha=0.7)
        
        if self.popt is not None:
            x_fit = np.linspace(min(self.x), max(self.x), 100)
            y_fit = self.model(x_fit, *self.popt)
            lower, upper = self.calculate_confidence_interval(x_fit)
            plt.fill_between(x_fit, lower, upper, color=line_color, alpha=0.2, label='95% Confidence Band')
            plt.plot(x_fit, y_fit, label='Fit', color=line_color, lw=2)
            
            # Diagnostic: Reduced Chi-Squared
            red_chi = self.calculate_reduced_chi_sq()
            eq_text = self.get_equation_string()
            
            stats_text = f"{eq_text}\n$\chi_\\nu^2 = {red_chi:.3f}$"
            plt.gca().text(
                0.80,
                0.05,
                stats_text,
                transform=plt.gca().transAxes,
                           verticalalignment='bottom',
                           horizontalalignment='right',
                           bbox=dict(boxstyle='round',
                           facecolor='white',
                           alpha=0.5))

        
        plt.title(title if title else "Curve Fit", fontsize=14, fontweight='bold')

        plt.xlabel(self.xaxis_label)
        plt.ylabel(self.yaxis_label)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

 # Sanity check
    def plot_residuals(self):
        residuals = self.calculate_residuals()
        if residuals is None: return

        plt.figure(figsize=(8, 4))
        plt.scatter(self.x, residuals, color='purple', marker='x')
        plt.axhline(0, color='black', linestyle='-')
        plt.xlabel(self.xaxis_label)
        plt.ylabel("Residuals")
        plt.title("Residual Analysis")
        plt.grid(True, alpha=0.3)
        plt.show()

class ExponentialFitter(BaseFitter):
    def __init__(self, x, y, xaxis_label='Time', yaxis_label='Counts', guess=(1, -0.1), background_subtract=None, dead_time=0, is_rate = False,
                t_gross = 1.0, t_bg = 5.0):
        super().__init__(x, y, xaxis_label, yaxis_label, background_subtract, dead_time, is_rate, t_gross, t_bg)
        self.guess = guess
        
        y_proc, sigma = self._process_data(linear_log=False)
        try:
            self.popt, self.pcov = cf(self.model, self.x, y_proc, p0=self.guess, sigma=sigma)
            self.perr = np.sqrt(np.diag(self.pcov))
        except Exception as e:
            print(f"Exponential Fit Error: {e}")
            
    def get_equation_string(self):
        a, b = self.popt
        return f"$y = {a:.2f} e^{{{b:.4f}x}}$"

    @staticmethod
    def model(x, a, b):
        return a * np.exp(b * x)

class LinearFitter(BaseFitter):
    def __init__(self, x, y, xaxis_label='x', yaxis_label='y',guess=(1,-0.1), background_subtract=None,dead_time=0):
        super().__init__(x, y, xaxis_label, yaxis_label, background_subtract, dead_time)
        self.use_log = True
        self.guess = guess
        y_proc, sigma = self._process_data(linear_log=True)
        try:
            # Linear model: m*x + c
            self.popt, self.pcov = cf(self.model, self.x, y_proc, p0=self.guess, sigma=sigma, absolute_sigma=True)
            self.perr = np.sqrt(np.diag(self.pcov))
        except Exception as e:
            print(f"Linear Fit Error: {e}")

    @staticmethod
    def model(x, m, c):
        return m * x + c
        
    def get_equation_string(self):
        m, c = self.popt
        return f"$y = {m:.4f}x + {c:.2f}$"
# Check to see how far from the mean    
def z_score(value, mean, std):
    return (value - mean) / std

class PopulationAnalyzer:
  def __init__(self, data, title, xaxis_label, yaxis_label, expected_mean, group, bins=5):
    self.data = data
    self.xaxis_label = xaxis_label
    self.yaxis_label = yaxis_label
    self.mean = np.mean(self.data) # lab population mean
    self.std = np.std(self.data, ddof=1) #lab std
    self.expected_mean = expected_mean #expected value
    self.title = title
    self.bins = bins # number of bars
    self.group = group # your data

  def ndf(self):
    lower_bound = self.mean - 4*self.std
    upper_bound = self.mean + 4*self.std
    x = np.linspace(lower_bound, upper_bound, 100)
    coefficient = 1/(np.sqrt(2*np.pi)*self.std)
    exponential = np.exp(-0.5*((x-self.mean)/ self.std)**2)
    y = coefficient * exponential
    return x, y
# Pool everyone's data
# Takes an array
  def plot_ndf(self):
    """Plots the normal distribution function"""
    x, y = self.ndf()
    group_y = (1/(np.sqrt(2*np.pi)*self.std)) * np.exp(-0.5*((self.group-self.mean)/ self.std)**2)
    plt.figure(figsize=(10, 6))

    # 1. Plot the smooth Gaussian curve
    plt.plot(x, y, color='black', label='Population Gaussian', linewidth=2)
    plt.plot(self.data, np.zeros_like(self.data), 'ro', alpha=0.5, markersize=4)
    # 2. Overlay the actual data points as a histogram
    # density=True makes the y-axis match the NDF probability scale
    plt.hist(self.data, bins=self.bins, density=True, alpha=0.15, color='blue', label='Group Results', edgecolor = 'black')
    x_curve, y_curve = self.ndf()
    plt.fill_between(x_curve, y_curve, where=(x_curve >= self.mean - self.std) & (x_curve <= self.mean + self.std),
                 color='purple', alpha=0.4, label='1-Sigma Range (68%)')
    
    plt.annotate('Our Group', xy=(self.group, group_y), xytext=(self.group, group_y+1),
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center')
    # 3. Add vertical lines for comparison
    plt.axvline(self.mean, color='green', linestyle='-', label=f'Lab Mean: {self.mean:.3f}')
    plt.axvline(self.expected_mean, color='red', linestyle='--', label=f'Expected: {self.expected_mean}')

    plt.xlabel(self.xaxis_label)
    plt.ylabel(self.yaxis_label)

    full_title = f"{self.title}\nN={len(self.data)}, Lab Mean={self.mean:.3f}, Std Dev={self.std:.3f}"
    plt.title(full_title)

    plt.legend()
    plt.grid(alpha=0.2)

    plt.show()






import matplotlib.pyplot as plt
plt.style.use('../styles/science.mplstyle')

def plot_validation(y_true,y_pred):
    """
    Plot predicted vs actual volatility on test data.
    """
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Time step')
    plt.ylabel('Volatility')
    plt.title('Predicted vs Actual Volatility')
    plt.legend()
    plt.show()
    # plt.savefig('../plots/predicted_vs_actual_volatility.pdf')

def plot_forecast(df,y_pred):
    """
    Plot forecasted volatility over future period.
    """
    plt.figure(figsize=(10,5))
    plt.plot(df['timestamp'].iloc[-len(y_pred):], y_pred, label='Forecasted Volatility')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.title('Forecasted Volatility Over Time')
    plt.legend()
    plt.show()
    # plt.savefig('../plots/forecasted_volatility.pdf')
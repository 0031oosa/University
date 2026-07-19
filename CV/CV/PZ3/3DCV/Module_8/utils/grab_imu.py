import pandas as pd


def custom(df: pd.DataFrame) -> pd.DataFrame:
    """
    timestamp,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z
    (timestamps=[ns], omega=[rad/s], alpha=[m/s^2])
    """
    col = ["timestamp",
           "omega_x",
           "omega_y",
           "omega_z",
           "alpha_x",
           "alpha_y",
           "alpha_z"]
    df.columns = col


if __name__ == "__main__":
    # Чтение файла
    df = pd.read_csv("gyro_accel.csv")
    custom(df)
    df.to_csv("./dataset/imu0.csv", index=False)
    print("done!")
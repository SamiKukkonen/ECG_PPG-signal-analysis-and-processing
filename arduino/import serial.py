import time
import numpy as np
import matplotlib.pyplot as plt

# Create empty lists to store data
timestamps_ecg = []
ecg_data = []
timestamps_ppg = []
ppg_data = []

# Create a figure and two axes for plotting (one for ECG and one for PPG)
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Set the initial update interval and maximum age of data in seconds
initial_update_interval = 500
max_age = 1000

# Open the file in read mode
with open('data.txt', 'r') as file:
    # Track the start time for clearing old data points
    start_time = time.time()
    last_plot_time = start_time
    update_interval = initial_update_interval  # Initialize update interval

    # Loop indefinitely
    while True:
        try:
            # Read a line from the file
            line = file.readline()

            # If the line is empty, wait for a moment and continue
            if not line:
                time.sleep(0.1)
                continue

            # Strip any leading/trailing whitespace and newline characters
            line = line.strip()

            # Split the line into parts (assuming comma-separated values)
            parts = line.split(',')

            # Check if the line has enough parts to represent ECG and PPG data
            if len(parts) < 4:
                print("Skipping line with invalid format:", line)
                continue

            # Attempt to extract ECG and PPG data
            try:
                # Assuming the first part is the PPG timestamp
                timestamp_ppg = float(parts[0].strip())
                ppg_value = float(parts[1].strip())
                # Assuming the third part is the ECG timestamp
                timestamp_ecg = float(parts[2].strip())
                ecg_value = float(parts[3].strip())
            except ValueError:
                print("Skipping line with non-numeric values:", line)
                continue

            # Append the data to the lists
            timestamps_ppg.append(timestamp_ppg)
            ppg_data.append(ppg_value)
            timestamps_ecg.append(timestamp_ecg)
            ecg_data.append(ecg_value)

            # Check if it's time to update the plot
            if len(timestamps_ecg) >= update_interval:
                # Plot the ECG data
                ax1.clear()  # Clear the previous plot
                ax1.plot(timestamps_ecg, ecg_data, label='ECG')
                ax1.set_ylabel('Amplitude (ECG)')
                ax1.legend()

                # Plot the PPG data
                ax2.clear()  # Clear the previous plot
                ax2.plot(timestamps_ppg, ppg_data, label='PPG')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Amplitude (PPG)')
                ax2.legend()

                # Show the plot
                plt.draw()
                plt.pause(0.001)  # Maintain the same update interval

                # Check if it's time to clear old data points
                if time.time() - start_time > max_age:
                    # Calculate the update interval based on the current number of data points
                    remaining_points = len(timestamps_ecg)
                    update_interval = max(remaining_points, initial_update_interval)

                    # Clear old data points
                    while timestamps_ecg and timestamps_ecg[0] < time.time() - max_age:
                        timestamps_ecg.pop(0)
                        ecg_data.pop(0)
                    while timestamps_ppg and timestamps_ppg[0] < time.time() - max_age:
                        timestamps_ppg.pop(0)
                        ppg_data.pop(0)

        except Exception as e:
            print("Error:", e)
            break  # Exit the loop when an exception occurs

# Close the plot window and exit
plt.close()

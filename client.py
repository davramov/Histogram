import argparse
import concurrent.futures
import cv2
import requests
from matplotlib import pyplot as plt
import numpy as np
from queue import Queue
from tabulate import tabulate
import time

class Client:
    """Client class for processing images and plotting histograms."""

    def __init__(self):
        """Initialize the client."""
        # Initialize a queue to store histograms
        self.queue = Queue()
        # Create a requests session to post messages to the histogram service
        self.session = requests.Session()

    def process_image_and_get_histogram(self, data, rgb_processing=False):
        """Process an image, calculate its histogram, and plot the histogram.

        Args:
            data (str or np.ndarray): Input data representing an image file path or a 2D numpy array.
            rgb_processing (bool): Flag indicating whether to request an RGB histogram.
        """
        filename = None
        if isinstance(data, str):
            filename = data
            # Read image data
            image_data = self.read_image(data)
            if image_data is None:
                return
            # Send image data to histogram service
            histogram_data = self.send_image_to_histogram_service(image_data, is_img=True, rgb_processing=rgb_processing)
        elif isinstance(data, np.ndarray):
            print("2D Array detected")
            # Send 2D numpy array data to histogram service
            histogram_data = self.send_image_to_histogram_service(data.tobytes(), is_img=False, rgb_processing=rgb_processing)
        else:
            print("Invalid input data. Please provide either an image file path or a valid 2D array.")

        # Process histogram data if available
        if 'histogram' in histogram_data:
            print(f"Image name: {filename}")
            histogram = histogram_data['histogram']
            metadata = [
                ["Number of bins", len(histogram)],
                ["Maximum bin value", max(histogram)],
                ["Minimum bin value", min(histogram)],
                ["Mean bin value", np.mean(histogram)],
                ["Median bin value", np.median(histogram)],
                ["Standard deviation of bin values", np.std(histogram)],
                ["Histogram range", f"{min(histogram)} to {max(histogram)}"],
                ["Peak value", np.argmax(histogram)],
                ["Total count or sum of bin values", sum(histogram)],
            ]
            headers = ["Metadata", "Value"]
            print(tabulate(metadata, headers=headers, tablefmt="grid"))

            # Put histogram data into the queue
            self.queue.put((histogram, metadata, filename))

        elif 'histograms' in histogram_data:
            print(f"Image name: {filename}")
            rgb_histograms = histogram_data['histograms']
            metadata = [
                ["Number of channels", len(rgb_histograms)],
                ["Number of bins per channel", len(rgb_histograms[0])],  # Assuming all channels have the same number of bins
                ["Peak value per channel", [np.argmax(channel) for channel in rgb_histograms]],
            ]
            headers = ["Metadata", "Value"]
            print(tabulate(metadata, headers=headers, tablefmt="grid"))

            # Put RGB histograms into the queue
            self.queue.put((rgb_histograms, metadata, filename))
    
    def read_image(self, filepath):
        """Read image data from disk.

        Args:
            filepath (str): Path to the image file.

        Returns:
            bytes: Image data read from the file.
        """
        try:
            # Read image from disk
            with open(filepath, 'rb') as f:
                image_data = f.read()
            return image_data
        except Exception as e:
            print("Error reading image:", e)
            return None
    
    def send_image_to_histogram_service(self, data, is_img, rgb_processing=False):
        """Send image data to the histogram service and receive histogram data.

        Args:
            data (bytes): Image data to be sent to the service.
            is_img (bool): Flag indicating if the data is loaded from an image file.
            rgb_processing (bool): Flag indicating whether to request an RGB histogram.

        Returns:
            dict: Histogram data received from the service.
        """
        try:
            # Determine the URL based on the type of data and RGB processing flag
            if rgb_processing:
                url = 'http://localhost:5000/process_image_and_calculate_RGB_histogram'
            else:
                url = 'http://localhost:5000/process_image_and_calculate_histogram' if is_img else 'http://localhost:5000/calculate_histogram'
            files = {'file': data}
            # Send image data to histogram service using the session
            response = self.session.post(url, files=files)
            response.raise_for_status()  # Raise an error for non-200 status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            print("Error sending image to service:", e)
            return {'error': 'Failed to send image to service'}
            
    def plot_histogram(self, histograms, metadata, filename=None):
        """Plot the histogram using Matplotlib with a dark theme.

        Args:
            histograms (list or dict): List or dictionary containing histogram data.
            metadata (list): List of metadata to be displayed below the plot.
            filename (str, optional): Filename of the image. Defaults to None.
        """
        try:
            # Set the dark theme
            plt.style.use('dark_background')

            # Set the size of the figure
            if filename:
                plt.figure(figsize=(12, 6))
            else:
                plt.figure(figsize=(8, 6))

            # Load thumbnail of the original image
            if filename:
                thumbnail = cv2.imread(filename)
                thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 2, 1)
                plt.imshow(thumbnail)
                plt.axis('off')
                plt.title(f'{filename}')

            # Plot the histograms using Matplotlib
            if filename:
                plt.subplot(1, 2, 2)

            # Define custom colors for each channel
            custom_colors = {'RED': 'r', 'GREEN': 'g', 'BLUE': 'b', 'GRAY': 'grey'}

            # Check if histograms contain multiple channels (RGB)
            if args.rgb is True:
                print("multichannel")
                # Multiple histograms (RGB)
                bar_width = 1  # Adjust the width of the bars
                for i, histogram in enumerate(histograms):
                    label = list(custom_colors.keys())[i] if i < len(custom_colors) else 'GRAY'
                    x = np.arange(len(histogram)) + i * (bar_width + 0.1)  # Add a small gap between bars
                    plt.bar(x, histogram, width=bar_width, label=label, color=custom_colors[label], alpha=0.75 if label == 'RED' or label == 'BLUE' or label == "GREEN" else 0.5)

                    # Draw a line plot over the box plot with the same color
                    plt.plot(x, histogram, color=custom_colors[label], alpha=0.8, linewidth=0.5)
                plt.legend()
            else:
                print("single channel")
                # Single histogram
                histograms = [histograms]  # Convert to a list to ensure it's iterable
                label = 'GRAY'
                num_bins = len(histograms[0])  # Assuming the histogram has 'num_bins' bins
                bar_width = 1  # Adjust the width of the bars
                x = np.arange(num_bins)  # Generate x values for each bin
                plt.bar(x, histograms[0], width=bar_width, color=custom_colors['GRAY'], label=label)

            # Plot labels
            plt.xlabel('Bins', fontsize=12, fontweight='bold')  # Customize x-axis label
            plt.ylabel('Frequency', fontsize=12, fontweight='bold')  # Customize y-axis label
            if filename:
                plt.title(f'Histogram - {filename}', fontsize=14, fontweight='bold')  # Customize title
            else:
                plt.title('Histogram', fontsize=14, fontweight='bold')  # Customize title

            plt.xticks(fontsize=10)  # Customize x-axis tick labels font size
            plt.yticks(fontsize=10)  # Customize y-axis tick labels font size
            plt.grid(True, linestyle='--', alpha=0.5)  # Add grid lines with dashed style and transparency

            # Add text annotations for metadata below the plot
            text = "\n".join([f"{row[0]}: {row[1]}" for row in metadata])
            plt.text(0.5, 0.05, text, fontsize=10, horizontalalignment='left', verticalalignment='bottom')

            if filename:
                plt.tight_layout()  # Adjust layout for better spacing
            plt.show()
        except Exception as e:
            print("Error plotting histogram:", e)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process images and plot histograms")
    parser.add_argument('data', nargs='+', help="Filepath strings or 2D numpy arrays")
    parser.add_argument('--rgb', action='store_true', help="Request RGB histogram")
    parser.add_argument('--s', action='store_true', help="Save the resulting histogram plot")
    args = parser.parse_args()

    # Initialize the client
    client = Client()

    # Function to process each image
    def process_image(data):
        client.process_image_and_get_histogram(data, rgb_processing=args.rgb)

    # Start timer for the program
    start_time_program = time.time()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each image for processing
        executor.map(process_image, args.data)

    # End timer for processing images
    end_time_processing = time.time()
    print(f"Time taken for processing images: {end_time_processing - start_time_program} seconds")

    # Plot histograms from the main thread
    while not client.queue.empty():
        histograms, metadata, filename = client.queue.get()
        client.plot_histogram(histograms, metadata, filename)
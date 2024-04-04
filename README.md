# Image Histogram Service

This service is designed to process 2D data files (image frames), calculate histograms, and handle responses from multiple clients concurrently.

## Installation

To install the dependencies from the `requirements.txt` file, use pip:

`pip install -r requirements.txt` 

## Usage

### Running the Service

To start the service, run the following command:

`python histogram_service.py` 

This command initializes the Quart application and starts the histogram service.

### Running the Client

To interact with the service and process images, run the client script:

`python client.py` 

The client script communicates with the service, sending image data for histogram calculation and processing the returned histogram data.

## Functionality


### Client

The Client class is responsible for processing images, sending them to the HistogramService, receiving histogram data, and plotting histograms.

#### Dependencies

-   `argparse`: For parsing command-line arguments.
-   `concurrent.futures`: For executing tasks concurrently.
-   `cv2` (OpenCV): For reading images.
-   `requests`: For making HTTP requests to the HistogramService.
-   `matplotlib.pyplot`: For plotting histograms.
-   `numpy`: For numerical operations.
-   `Queue`: For managing a queue of histogram data.
-   `tabulate`: For formatting metadata in tabular form.
-   `time`: For timing the execution of tasks.

#### Usage

To use the Client, execute the `client.py` script from the command line and provide the file paths of the images to be processed as arguments. Optionally, specify the `--rgb` flag to request RGB histograms or the `--s` flag to save the resulting histogram plots.

#### Functionality

-   `process_image_and_get_histogram(data, rgb_processing=False)`: Processes an image, sends it to the HistogramService, receives histogram data and adds it to the queue.
-   `read_image(filepath)`: Reads image data from disk.
-   `send_image_to_histogram_service(data, is_img, rgb_processing=False)`: Sends image data to the HistogramService and receives histogram data.
-   `plot_histogram(histograms, metadata, filename=None)`: Plots the histogram using Matplotlib.

### HistogramService

The HistogramService class is a Quart application that handles HTTP requests for calculating and processing image histograms.

#### Dependencies

-   `cv2` (OpenCV): For image processing.
-   `cupy`: For GPU-accelerated histogram calculations.
-   `quart`: For creating the asynchronous web service.
-   `quart_compress`: For compressing Quart responses.
-   `numpy`: For numerical operations.
-   `time`: For timing the execution of tasks.

#### Usage

To use the HistogramService, execute the `histogram_service.py` script from the command line.

#### Endpoints

-   `/calculate_histogram`: Endpoint for calculating histograms from a square 2D array.
-   `/process_image_and_calculate_histogram`: Endpoint for processing images and calculating histograms.
-   `/process_image_and_calculate_RGB_histogram`: Endpoint for processing images and calculating RGB histograms.

#### Functionality

-   `calculate_histogram(data)`: Calculates the histogram of an image.
-   `process_image_and_calculate_histogram(file)`: Processes an image file and calculates its histogram.
-   `process_image_and_calculate_RGB_histograms(file)`: Processes an image file and calculates its histograms for each RGB channel and grayscale.
-   `handle_calculate_histogram()`: Handler for calculating histograms from a square 2D array.
-   `handle_process_image_and_calculate_histogram()`: Handler for processing images and calculating histograms.
-   `handle_process_image_and_calculate_RGB_histogram()`: Handler for processing images and calculating RGB histograms.
-   `run()`: Runs the Quart application and defines the HTTP endpoints.

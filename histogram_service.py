import cv2
import cupy
from quart import Quart, request, jsonify
from quart_compress import Compress
import numpy as np
import time

class HistogramService:
    """Service class for calculating and processing image histograms."""

    def __init__(self):
        """Initialize the Quart application."""
        self.app = Quart(__name__)
        Compress(self.app)

    async def calculate_histogram(self, data):
        """Calculate the histogram of an image.

        Args:
            data (cupy.ndarray): Image data.

        Returns:
            list: Flattened histogram values.
        """
        try:
            start_time = time.time()
            bins = 256
            # Transfer the image data to the GPU memory
            hist = cupy.histogram(data, bins=bins, range=(0, 256))
            end_time = time.time()
            print(f"Time taken for calculating histogram: {end_time - start_time} seconds")
            return hist[0].tolist()  # Extract histogram values and convert to list
        except Exception as e:
            return {'error': str(e)}

    async def process_image_and_calculate_histogram(self, file):
        """Process an image file and calculate its histogram.

        Args:
            file (FileStorage): Uploaded image file.

        Returns:
            dict: Histogram data or error message.
        """
        try:
            start_time = time.time()
            np_data = np.frombuffer(file.read(), dtype=np.uint8)
            data = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            histogram = await self.calculate_histogram(cupy.asarray(data))
            end_time = time.time()
            print(f"Time taken for processing image and calculating histogram: {end_time - start_time} seconds")
            return {'histogram': histogram}
        except Exception as e:
            return {'error': str(e)}
    
    async def process_image_and_calculate_RGB_histograms(self, file):
        """Process an image file and calculate its histograms for each RGB channel and grayscale.

        Args:
            file (FileStorage): Uploaded image file.

        Returns:
            dict: Histogram data or error message.
        """
        try:
            start_time = time.time()
            np_data = np.frombuffer(file.read(), dtype=np.uint8)
            data = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            
            # Separate RGB channels
            channels = cv2.split(data)
            
            # Calculate histogram for each RGB channel
            histograms = []
            for channel in channels:
                histogram = await self.calculate_histogram(cupy.asarray(channel))
                histograms.append(histogram)
            
            # Convert image to grayscale
            grayscale = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram for grayscale image
            grayscale_histogram = await self.calculate_histogram(cupy.asarray(grayscale))
            histograms.append(grayscale_histogram)
            
            end_time = time.time()
            print(f"Time taken for processing image and calculating histograms: {end_time - start_time} seconds")
            
            return {'histograms': histograms}
        except Exception as e:
            return {'error': str(e)}

    async def handle_calculate_histogram(self):
        """Handler for calculating histogram from a square 2D Array."""
        try:
            start_time = time.time()
            if 'file' not in (await request.files):
                return jsonify({'error': 'No file part'})
            file = (await request.files)['file']
            np_data = np.frombuffer(file.read(), dtype=np.uint8)
            np_data = np_data.reshape((int(np.sqrt(len(np_data) / 3)), int(np.sqrt(len(np_data) / 3)), 3))
            np_data = cupy.asarray(np_data)
            histogram = await self.calculate_histogram(np_data)
            end_time = time.time()
            print(f"Time taken for handling calculate histogram request: {end_time - start_time} seconds")
            return jsonify({'histogram': histogram})
        except Exception as e:
            return jsonify({'error': str(e)})

    async def handle_process_image_and_calculate_histogram(self):
        """Handler for processing image and calculating histogram."""
        try:
            start_time = time.time()
            if 'file' not in (await request.files):
                return jsonify({'error': 'No file part'})
            file = (await request.files)['file']
            result = await self.process_image_and_calculate_histogram(file)
            end_time = time.time()
            print(f"Time taken for handling calculate histogram request: {end_time - start_time} seconds")
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
        
    async def handle_process_image_and_calculate_RGB_histogram(self):
        """Handler for processing image and calculating RGB histograms."""
        try:
            if 'file' not in (await request.files):
                return jsonify({'error': 'No file part'})
            file = (await request.files)['file']
            result = await self.process_image_and_calculate_RGB_histograms(file)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})

    def run(self):
        """Run the Quart application."""
        
        @self.app.route('/calculate_histogram', methods=['POST'])
        async def handle_calculate_histogram():
            return await self.handle_calculate_histogram()

        @self.app.route('/process_image_and_calculate_histogram', methods=['POST'])
        async def handle_process_image_and_calculate_histogram():
            return await self.handle_process_image_and_calculate_histogram()

        @self.app.route('/process_image_and_calculate_RGB_histogram', methods=['POST'])
        async def handle_process_image_and_calculate_RGB_histogram():
            return await self.handle_process_image_and_calculate_RGB_histogram()

        self.app.run(debug=False)

if __name__ == '__main__':
    # Start the histogram service
    service = HistogramService()
    service.run()

# Self-Checkout System

This project implements a self-checkout system using two USB cameras and object detection.

## Setup

1. Clone the repository:

git clone https://github.com/your-username/self-checkout-system.git
cd self-checkout-system


## Running with Docker

- Make sure you have Docker and Docker Compose installed on your system.

- Build and run the Docker containers:

docker-compose up --build

- The application will be available at `http://localhost:5000`.

- To stop the containers, use:

docker-compose down


2. Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. Install the required packages:
pip install -r requirements.txt

4. Download YOLO weights and configuration:
- Download `yolovx.weights` from the official YOLO website
- Place `yolovx.weights` in the `config/` directory

5. Perform camera calibration:
python calibration/calibration.py
python calibration/stereo_calibration.py

6. Run the main application:
python src/main.py

## API Endpoints

- GET `/health`: Health check endpoint
- POST `/checkout`: Perform checkout process

## Running Tests

To run the unit tests:
pytest tests/


## Project Structure

- `calibration/`: Camera calibration scripts and images
- `config/`: YOLO configuration files
- `src/`: Main source code
- `tests/`: Unit tests
- `data/`: Calibration results and other data

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
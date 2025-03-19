# Flask MySQL Web Application

A simple web application with user authentication and CRUD operations using Flask and MySQL.

## Features

1. **User Registration and Login**
   - Secure user authentication system
   - Password hashing for security
   - User session management

2. **CRUD Operations**
   - Create, read, update, and delete records in the database
   - User-specific item management
   - Data validation

3. **Web Interface**
   - Responsive design using Bootstrap
   - User-friendly forms
   - Flash messages for user feedback

4. **Security Features**
   - Password hashing with bcrypt
   - CSRF protection
   - Form validation
   - Secure user sessions

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- MySQL Server
- pip (Python package manager)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd flask-mysql-app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory with the following variables:
   ```
   SECRET_KEY=your_secret_key
   DATABASE_URL=mysql://username:password@localhost/flask_app
   ```

5. **Create the MySQL database**
   
   Log in to MySQL and create a new database:
   ```sql
   CREATE DATABASE flask_app;
   ```

6. **Initialize the database**
   
   Run the following commands to set up the database tables:
   ```bash
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

7. **Run the application**
   ```bash
   python run.py
   ```

8. **Access the application**
   
   Open your web browser and navigate to `http://localhost:5000`

## Usage Guide

### User Registration and Login

1. Click on the "Register" link in the navigation bar
2. Fill out the registration form with your username, email, and password
3. Submit the form to create your account
4. Log in using your email and password

### Managing Items

1. **Create a new item**
   - Click on "New Item" in the navigation bar
   - Fill out the item form with a title and description
   - Click "Submit" to create the item

2. **View items**
   - All items are displayed on the home page
   - Click on "View Details" to see the full item

3. **Update an item**
   - Navigate to the item details page
   - Click "Update" to edit the item
   - Make your changes and click "Submit"

4. **Delete an item**
   - Navigate to the item details page
   - Click "Delete" to remove the item
   - Confirm the deletion in the modal dialog

### User Profile

- Click on your username in any item to see all items created by that user

## Error Handling

The application includes comprehensive error handling:

- Form validation errors are displayed inline
- Flash messages provide feedback on actions
- 404 and 403 errors are handled gracefully

## Security Considerations

- Passwords are hashed using bcrypt before storage
- CSRF protection is enabled for all forms
- User input is validated to prevent injection attacks
- User sessions are managed securely

## Development

To run the application in development mode with debug enabled:

```bash
python run.py
```

## Testing

Run the tests using pytest:

```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

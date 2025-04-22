CREATE DATABASE SmartFile;
USE SmartFile;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,  -- Removed NOT NULL to allow training data
    filename VARCHAR(255) NOT NULL,
    folder VARCHAR(255) NOT NULL,
    text LONGTEXT,
    tables JSON,
    file_path VARCHAR(512) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FULLTEXT (text)
);

CREATE TABLE feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_id INT NOT NULL,
    user_id INT NOT NULL,
    predicted_folder VARCHAR(255) NOT NULL,
    corrected_folder VARCHAR(255),
    feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES files(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

ALTER TABLE users RENAME COLUMN username to email;
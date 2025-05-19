CREATE DATABASE smartFile;
USE smartFIle;

CREATE TABLE users ( 
    id INT AUTO_INCREMENT PRIMARY KEY, 
    email VARCHAR(255) UNIQUE NOT NULL, 
    password VARCHAR(255) NOT NULL 
);

CREATE TABLE files ( 
    id INT AUTO_INCREMENT PRIMARY KEY, 
    user_id INT, filename VARCHAR(255) NOT NULL, 
    folder VARCHAR(255) NOT NULL, 
    text TEXT, 
    tables TEXT, 
    file_path VARCHAR(512) NOT NULL, 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE 
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
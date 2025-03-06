### Software Engineering Project Proposal
# Title: ClassInsights - Classroom Analytics Platform

### Overview: 
**ClassInsights** would be an innovative classroom analytics tool designed to monitor student participation and attention in a real classroom setting. By leveraging computer vision and machine learning, this platform aims to automate attendance tracking, monitor student engagement, generate detailed subject-wise analytics and personal analytics of each student.
The application would take the live classroom feed as video input and then process the video to generate these insights, and this data is then saved in a database and can be accessed through the main dashboard.

### Feature List:
* Automated Attendance Sheet Generation using facial recognition techniques.
* Interaction and Attention Monitoring using parameters such as PERCLOS score for drowsiness detection and head-pose estimation to detect loss of interest or lack of attention.
* Student Profile Management Utility to add new student data to the database.
* Analytics Dashboard to display and manage attendance records for each subject, generate an engagement score for each subject and student.

### Implementation Plan:

1. **Real Time Face Detection:**
This can be used to generate the attendance report and generate the trends of attendance for each subject.
We can use robust facial-recognition libraries to generate the image embeddings of the face and then match them against the image embeddings in the database to detect the presence of a student

2. **Interaction and Engagement Monitoring:**
Drowsiness Detection Using PERCLOS parameter:
The PERCLOS score is used by many modern vehicles to detect if the driver is drowsy or tired. 
It works by calculating the percentage amount of time the the eyes of an individual are closed, using which we can monitor the alertness of each student and of the class.
Attention Monitoring using Head Pose Estimation:
We can extract the facial landmarks of the student and then map them into a mesh, using which we can then calculate the yaw, pitch and roll determining if a student is looking sideways and is distracted frequently or is looking down disinterested in the class

3. **Student Profile Management Utility:**
The application is also aimed to have a system to add and remove student and subject data
The student face scan data would be converted into image embeddings and stored in a database along with the relevant information like roll number, department, etc.
We may use tools like SQLite to store and query the database efficiently, and methods such as KNN (K-nearest neighbor) to efficiently query the image embeddings for facial recognition.

4. **Analytics Dashboard:**
   * **Data Aggregation:**
   We can aggregate and structure the raw data such as attendance records, class interactive scores and other metrics to display on the dashboard

### Conclusion:
ClassInsights stands at the intersection of education and technology, offering a powerful solution to monitor and improve classroom dynamics. By integrating real-time face recognition, comprehensive student profile management, and interaction monitoring techniques, the platform provides educators with actionable insights into attendance and engagement. The implementation plan utilizes established Python libraries and frameworks, ensuring a modular, scalable, and robust system that can adapt to both online and physical classroom environments.

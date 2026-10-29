pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t churn-app:latest .'
            }
        }
        stage('Deploy') {
            steps {
                sh 'docker rm -f churn || true'
                sh 'docker run -d --name churn -p 8501:8501 churn-app:latest'
            }
        }
    }
}

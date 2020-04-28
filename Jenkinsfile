pipeline {
  agent any
  stages {
    stage('flake8') {
      steps {
        sh 'echo "hi"'
        sh 'pwd'
        sh 'flake8'
      }
    }

  }
}
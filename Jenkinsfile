node {
    stage('Git') {
        git branch: 'main', credentialsId: 'GitlabDavid', url: 'https://gitlab.com/DLefrancq/ad_project_api'
    }
    stage('Build') {
        image = docker.build("virtus/david:ad_project_api", "-f Dockerfile .")
        sh 'docker images'
    }
    stage('Deploy') {
        sh 'docker login -u virtus -p $DOCKER_PASSWORD'
        sh 'docker push virtus/david:ad_project_api'
    }
    stage('Clean') {
        sh 'docker rmi virtus/david:ad_project_api'
        sh 'docker images'
    }
    stage('Depmloiment CD') {
        ansiblePlaybook(credentialsId: 'test', inventory: 'ansible/inventories/hosts', playbook: 'cdihm.yml')
    }
}
# DroneMapping
Gets a map using a video captured from a Drone flight.

![lenta](https://user-images.githubusercontent.com/43963246/98108725-c132de80-1e9c-11eb-9f71-c487cdc3597b.jpg)



# Installation and run

To run this application you need OpenCV (version 4.3.0 minimun required) and CMake (version 3.5.1 minimun required).
From your command line:

$ git clone git@github.com:AdrianLopezGue/daruma-backend.git

# Go into the repository
$ cd daruma-backend

# Install dependencies
$ npm install

# Start docker containers (MongoDB and Event Store)
$ docker-compose up  -d

# Run the server app
$ npm run start:prod

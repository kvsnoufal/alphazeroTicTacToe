docker build --tag tictactoebackend .


docker images

docker tag tictactoebackend:latest tictactoebackend:v1.0.0

docker run -p 5000:5000 tictactoebackend


heroku container:login
 #heroku create <name-for-your-app>
heroku container:push web --app tictactoealphazero
heroku container:release web --app tictactoealphazero



docker build --tag tictactoefrontend .


docker run -p 3001:3000 tictactoefrontend

docker ps
docker stop goofy_leavitt


<!-- heroku create alphazerotictactoe -->
heroku container:push web --app alphazerotictactoe

heroku container:release web --app alphazerotictactoe
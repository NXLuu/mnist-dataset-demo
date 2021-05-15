const express = require('express')
const app = express()

app.use(function(req, res, next) {
    console.log("1");
    next();
})
app.use(express.static('./public'))

app.get('/', (req, res) => {
  res.render('room')
})


app.listen(3000);
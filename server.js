//express import
const exp=require("express")
const  app=exp()

//import userAPI
const userAPP=require("./APIs/userAPI")

//mongoClient import
const mclient=require('mongodb').MongoClient

//connect mclilent to monogodb server
mclient.connect('mongodb://localhost:27017')
.then(dbRef=>{
    let dbObj=dbRef.db('demodb')

    //create Collection object
    let userCollection=dbObj.createCollection('userCollection')

    //share collection object to API
    app.set('userCollection',userCollection)

    console.log("connected to database")
})
.catch(err=>console.log("error is",err))

//forward request to api
app.use('/user-api',userAPP)


//adding port number
app.listen(4000,()=>{console.log("server listening on port number 4000...")})

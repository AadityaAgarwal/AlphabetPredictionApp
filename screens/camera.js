import * as React from 'react'
import { Button,View,Image,Platform } from 'react-native'
import * as ImagePicker from 'expo-image-picker'
import * as Permissions from 'expo-permissions'

export default class PickImage extends React.Component{
    state={
        image:null,
    }

    get_permissions=async()=>{
        if (Platform.OS!="web"){
            const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
            if (status !== 'granted') {
              alert('Sorry, we need camera roll permissions to make this work!');
            }
        }
    }
    uploadImage=async(uri)=>{
        const data=new FormData();
        let filename=uri.split("/")[uri.split("/").length-1]
        let type=`image/${uri.split('.')[uri.split('.').length-1]}`
        const fileToUpload={
            uri:uri,
            name:filename,
            type:type,
        };
        data.append("digit",fileToUpload);
        fetch(' http://c90d86bb94c8.ngrok.io/predictAlphabet',{
            method:'POST',
            body:data,
            headers:{"content-type":"multipart/form-data,"},

        })
        .then((response)=>response.json())
        .then((result)=>{console.log("Success: ",result)})
        .catch((error)=>{console.error("Error: ",error)})
    }

    pick_image=async()=>{
        try{
            let result=await ImagePicker.launchImageLibraryAsync({
                mediaTypes:ImagePicker.MediaTypeOptions.All,
                allowsEditing:true,
                aspect:[4,3],
                quality:1,
            })
            if (!result.cancelled){
                this.setState({image:result.data})
                console.log(result.uri)
                this.uploadImage(result.uri)
            }
        }
        catch(error){
            console.log(error)
        }
    }

    componentDidMount(){
        this.get_permissions()
    }

    render(){
        let {image}=this.state
        return(
            <View style={{alignItems:'center',justifyContent:'center',flex:1}}>
                <Button title='Pick an image!' 
                onPress={
                    this.pick_image
                }/>
            </View>
        )
    }
}
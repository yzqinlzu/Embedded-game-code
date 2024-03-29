
(function(){
"use strict";

Recorder.prototype.enc_mp3={
	stable:true
	,testmsg:"采样率范围48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000"
};


Recorder.prototype.mp3=function(res,True,False){
		var This=this,set=This.set,size=res.length;
		//https://github.com/wangpengfei15975/recorder.js
		//https://github.com/zhuker/lamejs bug:采样率必须和源一致，不然8k时没有声音，有问题fix：https://github.com/zhuker/lamejs/pull/11
		var mp3=new Recorder.lamejs.Mp3Encoder(1,set.sampleRate,set.bitRate);
		
		var blockSize=5760;
		var data=[];
		
		var idx=0;
		var run=function(){
			if(idx<size){
				var buf=mp3.encodeBuffer(res.subarray(idx,idx+blockSize));
				if(buf.length>0){
					data.push(buf);
				};
				idx+=blockSize;
				setTimeout(run);//Worker? 复杂了
			}else{
				var buf=mp3.flush();
				if(buf.length>0){
					data.push(buf);
				};
				
				True(new Blob(data,{type:"audio/mp3"}));
			};
		};
		run();
	}
	
})();
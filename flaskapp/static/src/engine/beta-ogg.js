
(function(){
"use strict";

Recorder.prototype.enc_ogg={
	stable:false
	,testmsg:"比特率16-100kbps，此编码器源码2.2M，超级大，压缩后1.6M，开启gzip后327K左右，对录音的压缩率非常出色(相对mp3)"
};
Recorder.prototype.ogg=function(res,True,False){
		var This=this,set=This.set,size=res.length,bitRate=set.bitRate;
		bitRate=Math.min(Math.max(bitRate,16),100);
		set.bitRate=bitRate;
		
		bitRate=Math.max(1.1*(bitRate-16)/(100-16)-0.1, -0.1);//取值-0.1-1，实际输出16-100kbps
		var ogg = new Recorder.OggVorbisEncoder(set.sampleRate, 1, bitRate);
		
		var blockSize=set.sampleRate;
		
		var idx=0;
		var run=function(){
			if(idx<size){
				var buf=res.subarray(idx,idx+blockSize);
				var floatBuf=new Float32Array(set.sampleRate);
				for(var j=0;j<size;j++){
					var s=buf[j];
					s=s<0?s/0x8000:s/0x7FFF;
					floatBuf[j]=s;
				};
				ogg.encode([floatBuf]);
				
				idx+=blockSize;
				setTimeout(run);//Worker? 复杂了
			}else{
				True(ogg.finish("audio/ogg"));
			};
		};
		run();
	}
	
})();
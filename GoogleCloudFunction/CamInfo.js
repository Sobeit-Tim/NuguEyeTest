/**
 * Responds to any HTTP request.
 *
 * @param {!express:Request} req HTTP request context.
 * @param {!express:Response} res HTTP response context.
 */

exports.CamInfo = (req, res) => {
  const requestBody = req.body;
  const HOST = '34.97.244.124';
  const PORT = 10080
  if(requestBody.hasOwnProperty('action')) 
  {
    // 누구 디바이스에서 온 명령
    let parameters = requestBody.action.parameters;
    const context = requestBody.action.context; //컨텍스트, OAuth연결시 토큰이 들어옵니다
    const actionName = requestBody.action.actionName; // action의 이름
    let output = {};
    function makeJson(jsons) {
      /**
       * [makeJson 설명]
       * @json {jsons}
       * 안에는 누구로 보낼 json들이 있습니다
       * json안에는 파라메터들이 있으며, 각 파라메터는 sk nugu의 play에서 지정한
       * 이름과 동일해야 합니다.
       */
  
      let jsonReturn = {
        "version": "2.0",
        "resultCode": "OK",
        "directives": [
		{
        "type": "Display.FullText1",
        "version": "1.0", // 필수
        "playServiceId": "",
        "token": "",
        "title" : {
          "logo": {
            "contentDescription": "",
            "sources": [
                {
                    "url": "https://s.pstatic.net/static/www/img/uit/2019/sp_search.svg"
                }
            ]
          },
          "text": {
              "text": "NUGU 백과"
          }
        },
        "background": {
          "image": {
            "contentDescription": "",
            "sources": [
                {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/5/5e/Dokdo_Photo.jpg",
                    "size": "LARGE"
                }
              ]
            },
          "color": ""
        },
        "content": {
          "header": {
            "text": "독도"
          },
          "body": {
            "text": "‘독도’는 동해의 남서부인 울릉도와 오키 제도 사이에 있는 섬으로, 동도와 서도를 포함하고 총 91개의 섬들로 이루어져 있습니다."
          },    
          "footer": {
              "text": "출처 : 위키피디아"
          }
        }
      }
	  ]
      }
      jsonReturn.output = jsons
      return jsonReturn;
    } //makeJson
    function msg(tmp)
    {
      console.log(tmp);
      let stat=Number(tmp.substring(0,1));
      let dis=Number(tmp.substring(16,tmp.length));
      let no =Number(tmp.substring(7,8));
      let left=parseFloat(tmp.substring(8,12));
      let right=parseFloat(tmp.substring(12,16));
      output.status=tmp.substring(0,1);
      output.eye=tmp.substring(1,6);
      output.answer=tmp.substring(6,7);
      output.no=tmp.substring(7,8);
      output.left=tmp.substring(8,12);
      output.right=tmp.substring(12,16);
      output.dis=tmp.substring(16,tmp.length);
      output.response=''
      if (stat == 0)
      {
        if (dis==2)
        {
          output.response = "왼쪽 눈부터 측정 하겠습니다 오른쪽 눈을 가려주시고 눈 가렸어 라고 말씀해주세요"
        }else if(dis < 2)
        {
          output.response = "현재 거리는 " + dis + "미터입니다. "+ (2-dis) +"미터 떨어져 주세요"
        }else
        {
          output.response = "현재 거리는 " + dis + "미터입니다. "+ (dis-2) +"미터 가까이 와주세요"
        }
      }else if(stat ==2)
      {
        if(no==3)
        {
           output.response = "왼쪽 눈 시력검사가 끝났습니다. 오른쪽 눈을 측정 하겠습니다. 왼쪽 눈을 가려주시고 눈 가렸어 라고 말씀해주세요"
        }else 
        {
          output.response = "다음 그림입니다"
        }
      }else if(stat==4)
      {
        if(no==3)
        {
          output.response="모든 시력 측정이 끝났습니다 왼쪽 눈은 "+ left.toString() + " 오른쪽 눈은 " + right.toString()+ " 입니다 시력 검사를 종료하려면 종료라고 말씀 해 주세요"
        }else
        {
          output.response = "다음 그림입니다"
        }
      }
    }
    function checkStart() {
      // 시력 검사 시작
      var net = require('net');
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(3000);
        this.setEncoding('utf8');
        client.write('start');
        //output.msg='서버가 응답없습니다'
        this.on('data', function(data) {
          tmp=data.toString();
          msg(tmp);
          return res.send(makeJson(output));
        });
      })
    }
	function checkDistance() {
      //거리 인식
      var net = require('net');
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(3000);
        this.setEncoding('utf8');
        client.write('distance');
        this.on('data', function(data) {
          tmp=data.toString();
          msg(tmp);
          client.end();
          return res.send(makeJson(output));
        });
      })  
    }
    function checkEye() {
      // 눈 확인
      var net = require('net');
      var status=0,eye='';
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(4000);
        this.setEncoding('utf8');
        client.write('eye');
        this.on('data', function(data) {
          tmp=data.toString();
          msg(tmp);
          client.end();
          return res.send(makeJson(output));
        });
      })  
    }
    function checkAnswer() {
      //오른쪽 왼쪽 대답
      var net = require('net');
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(2000);
        this.setEncoding('utf8');
        if (parameters.answer.value=='오른쪽')
        {
        	client.write('r');  
        }else if(parameters.answer.value=='왼쪽')
        {
          client.write('l');  
        }else if(parameters.answer.value=='위')
        {
          client.write('u');  
        }else if(parameters.answer.value=='아래')
        {
          client.write('d');  
        }
        
        this.on('data', function(data) {
          tmp=data.toString();
          msg(tmp);
          client.end();
          return res.send(makeJson(output));
        });
      })  
    }
    function exits() {
      // 시력검사 종료시켜서 서버 초기화 시키는 부분
      var net = require('net');
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(500);
        this.setEncoding('utf8');
        client.write('exit');
        client.end();
        return res.send(makeJson(output));
      })  
    }
    function takePic() {
      //오른쪽 왼쪽 대답
      var net = require('net');
      output.result = '못 찍었습니다'
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(1000);
        this.setEncoding('utf8');
        client.write('take');
        this.on('data', function(data) {
          if(data =='4')
          {
		    output.result = '찍었습니다'
            client.end();
            return res.send(makeJson(output));
          }
        });
      })  
    }
    
    function checkFace() {
      // 얼굴 인식
      var net = require('net');
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(5000);
        this.setEncoding('utf8');
        client.write('face');
        //output.msg='서버가 응답없습니다'
        this.on('data', function(data) {
          tmp=data.toString();
          msg(tmp);
          return res.send(makeJson(output));
        });
      })
    }
    
    //액션 선언 부분
    const ACTION_TAKE = 'takePic';
    const ACTION_START = 'checkStart';
    const ACTION_DISTANCE = 'checkDistance';
    const ACTION_EYE = 'checkEye';
    const ACTION_ANSWER = 'checkAnswer';
    const ACTION_FACE = 'checkFace';
    const ACTION_EXIT = '';
    
    // Intent가 오는 부분, actionName으로 구분합니다.
    // case안에서 작동할 function을 적습니다.
    switch (actionName) {
      case ACTION_TAKE:
        return takePic()
        break;
      case ACTION_START:
        return checkStart()
        break;
      case ACTION_DISTANCE:
        return checkDistance()
        break;
      case ACTION_EYE:
        return checkEye()
        break;
      case ACTION_ANSWER:
        return checkAnswer()
        break;
      case ACTION_FACE:
        return checkFace()
        break;
      case ACTION_EXIT:
        return exits()
        break;
    }
  }
};/**
 * Responds to any HTTP request.
 *
 * @param {!express:Request} req HTTP request context.
 * @param {!express:Response} res HTTP response context.
 */

exports.CamInfo = (req, res) => {
  const requestBody = req.body;
  const HOST = '34.97.244.124';
  const PORT = 10080
  if(requestBody.hasOwnProperty('action')) 
  {
    // 누구 디바이스에서 온 명령
    let parameters = requestBody.action.parameters;
    const context = requestBody.action.context; //컨텍스트, OAuth연결시 토큰이 들어옵니다
    const actionName = requestBody.action.actionName; // action의 이름
    let output = {};
    function makeJson(jsons) {
      /**
       * [makeJson 설명]
       * @json {jsons}
       * 안에는 누구로 보낼 json들이 있습니다
       * json안에는 파라메터들이 있으며, 각 파라메터는 sk nugu의 play에서 지정한
       * 이름과 동일해야 합니다.
       */
  
      let jsonReturn = {
        "version": "2.0",
        "resultCode": "OK",
        "directives": [
		{
        "type": "Display.FullText1",
        "version": "1.0", // 필수
        "playServiceId": "",
        "token": "",
        "title" : {
          "logo": {
            "contentDescription": "",
            "sources": [
                {
                    "url": "https://s.pstatic.net/static/www/img/uit/2019/sp_search.svg"
                }
            ]
          },
          "text": {
              "text": "NUGU 백과"
          }
        },
        "background": {
          "image": {
            "contentDescription": "",
            "sources": [
                {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/5/5e/Dokdo_Photo.jpg",
                    "size": "LARGE"
                }
              ]
            },
          "color": ""
        },
        "content": {
          "header": {
            "text": "독도"
          },
          "body": {
            "text": "‘독도’는 동해의 남서부인 울릉도와 오키 제도 사이에 있는 섬으로, 동도와 서도를 포함하고 총 91개의 섬들로 이루어져 있습니다."
          },    
          "footer": {
              "text": "출처 : 위키피디아"
          }
        }
      }
	  ]
      }
      jsonReturn.output = jsons
      return jsonReturn;
    } //makeJson
    function msg(tmp)
    {
      console.log(tmp);
      let stat=Number(tmp.substring(0,1));
      let dis=Number(tmp.substring(16,tmp.length));
      let no =Number(tmp.substring(7,8));
      let left=parseFloat(tmp.substring(8,12));
      let right=parseFloat(tmp.substring(12,16));
      output.status=tmp.substring(0,1);
      output.eye=tmp.substring(1,6);
      output.answer=tmp.substring(6,7);
      output.no=tmp.substring(7,8);
      output.left=tmp.substring(8,12);
      output.right=tmp.substring(12,16);
      output.dis=tmp.substring(16,tmp.length);
      output.response=''
      if (stat == 0)
      {
        if (dis==2)
        {
          output.response = "왼쪽 눈부터 측정 하겠습니다 오른쪽 눈을 가려주시고 눈 가렸어 라고 말씀해주세요"
        }else if(dis < 2)
        {
          output.response = "현재 거리는 " + dis + "미터입니다. "+ (2-dis) +"미터 떨어져 주세요"
        }else
        {
          output.response = "현재 거리는 " + dis + "미터입니다. "+ (dis-2) +"미터 가까이 와주세요"
        }
      }else if(stat ==2)
      {
        if(no==3)
        {
           output.response = "왼쪽 눈 시력검사가 끝났습니다. 오른쪽 눈을 측정 하겠습니다. 왼쪽 눈을 가려주시고 눈 가렸어 라고 말씀해주세요"
        }else 
        {
          output.response = "다음 그림입니다"
        }
      }else if(stat==4)
      {
        if(no==3)
        {
          output.response="모든 시력 측정이 끝났습니다 왼쪽 눈은 "+ left.toString() + " 오른쪽 눈은 " + right.toString()+ " 입니다 시력 검사를 종료하려면 종료라고 말씀 해 주세요"
        }else
        {
          output.response = "다음 그림입니다"
        }
      }
    }
    function checkStart() {
      // 시력 검사 시작
      var net = require('net');
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(3000);
        this.setEncoding('utf8');
        client.write('start');
        //output.msg='서버가 응답없습니다'
        this.on('data', function(data) {
          tmp=data.toString();
          msg(tmp);
          return res.send(makeJson(output));
        });
      })
    }
	function checkDistance() {
      //거리 인식
      var net = require('net');
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(3000);
        this.setEncoding('utf8');
        client.write('distance');
        this.on('data', function(data) {
          tmp=data.toString();
          msg(tmp);
          client.end();
          return res.send(makeJson(output));
        });
      })  
    }
    function checkEye() {
      // 눈 확인
      var net = require('net');
      var status=0,eye='';
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(4000);
        this.setEncoding('utf8');
        client.write('eye');
        this.on('data', function(data) {
          tmp=data.toString();
          msg(tmp);
          client.end();
          return res.send(makeJson(output));
        });
      })  
    }
    function checkAnswer() {
      //오른쪽 왼쪽 대답
      var net = require('net');
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(2000);
        this.setEncoding('utf8');
        if (parameters.answer.value=='오른쪽')
        {
        	client.write('r');  
        }else if(parameters.answer.value=='왼쪽')
        {
          client.write('l');  
        }else if(parameters.answer.value=='위')
        {
          client.write('u');  
        }else if(parameters.answer.value=='아래')
        {
          client.write('d');  
        }
        
        this.on('data', function(data) {
          tmp=data.toString();
          msg(tmp);
          client.end();
          return res.send(makeJson(output));
        });
      })  
    }
    function exits() {
      // 시력검사 종료시켜서 서버 초기화 시키는 부분
      var net = require('net');
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(500);
        this.setEncoding('utf8');
        client.write('exit');
        client.end();
        return res.send(makeJson(output));
      })  
    }
    function takePic() {
      //오른쪽 왼쪽 대답
      var net = require('net');
      output.result = '못 찍었습니다'
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(1000);
        this.setEncoding('utf8');
        client.write('take');
        this.on('data', function(data) {
          if(data =='4')
          {
		    output.result = '찍었습니다'
            client.end();
            return res.send(makeJson(output));
          }
        });
      })  
    }
    
    function checkFace() {
      // 얼굴 인식
      var net = require('net');
	  var client = net.connect({port: PORT, host: HOST},function(){
	    //net모듈의 소켓 객체를 사용
        this.setTimeout(5000);
        this.setEncoding('utf8');
        client.write('face');
        //output.msg='서버가 응답없습니다'
        this.on('data', function(data) {
          tmp=data.toString();
          msg(tmp);
          return res.send(makeJson(output));
        });
      })
    }
    
    //액션 선언 부분
    const ACTION_TAKE = 'takePic';
    const ACTION_START = 'checkStart';
    const ACTION_DISTANCE = 'checkDistance';
    const ACTION_EYE = 'checkEye';
    const ACTION_ANSWER = 'checkAnswer';
    const ACTION_FACE = 'checkFace';
    const ACTION_EXIT = '';
    
    // Intent가 오는 부분, actionName으로 구분합니다.
    // case안에서 작동할 function을 적습니다.
    switch (actionName) {
      case ACTION_TAKE:
        return takePic()
        break;
      case ACTION_START:
        return checkStart()
        break;
      case ACTION_DISTANCE:
        return checkDistance()
        break;
      case ACTION_EYE:
        return checkEye()
        break;
      case ACTION_ANSWER:
        return checkAnswer()
        break;
      case ACTION_FACE:
        return checkFace()
        break;
      case ACTION_EXIT:
        return exits()
        break;
    }
  }
};

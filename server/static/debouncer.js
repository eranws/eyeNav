(function(){
  var lock = false; 
  $('body').on("down", function(){
    keyDown('down');
  }).on("right", function(){
    keyDown('right');
  }).on("up", function(){
    keyDown('up');
  }).on("left", function(){
    keyDown('left');
  });

  var keyDown = function(key){
  	if (!lock){
  		$('body').trigger(key+'2048');
  		lock = true;
  		setTimeout(function(){
  			lock = false;
  		},500);
  	}
  }
})()
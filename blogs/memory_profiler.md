#Python如何做内存监控-memory_profiler

先通过三个步骤

- 第一步下载安装你的memory_profiler
		
	<pre>pip install git+https://github.com/pythonprofilers/memory_profiler.git </pre>

- 第二步在需要监控的python脚本里导入
	
	<pre>#from memory_profiler import profile </pre>
- 第三步在你需要监控的函数上面加一个装饰器
	<pre>
    @profile
	def forward(self, sentence):
        embeds = self.word_embeddings(sentence)

        x = embeds.view(sentence.size()[1], self.batch_size, -1)
        self.hidden= self.init_hidden()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        return y
	</pre>

	给一个整体的例子哈， 见[test.py](test.py)
	<pre>
	from memory_profiler import profile
	@profile
	def my_func():
	    a = [1] * (10 ** 6)
	    b = [2] * (2 * 10 ** 7)
	    del b
	    return a		 
	if __name__ == '__main__':
	    my_func()
	</pre>

import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Input } from "@/components/ui/input";
import BotIcon from '@/components/ui/bot-icon';
import LoaderIcon from '@/components/ui/loader-icon';
import styles from './ChatInterface.module.css';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  thinkingProcess?: {
    conversationalStage: string,
    useTools: boolean,
    tool?: string,
    toolInput?: string,
    actionOutput?: string,
    actionInput?: string
  };
};

type ThinkingProcess = {
  conversationalStage: string,
  tool?: string,
  toolInput?: string,
  actionOutput?: string,
  actionInput?: string
};

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [thinkingProcess, setThinkingProcess] = useState<ThinkingProcess[]>([]);
  const [maxHeight, setMaxHeight] = useState('80vh');
  const [isBotTyping, setIsBotTyping] = useState(false);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  const thinkingProcessEndRef = useRef<null | HTMLDivElement>(null);
  const [botHasResponded, setBotHasResponded] = useState(false);

  useEffect(() => {
    const handleResize = () => setMaxHeight(`${window.innerHeight - 200}px`);
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    thinkingProcessEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [thinkingProcess]);

  useEffect(() => {
    if (botHasResponded) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      thinkingProcessEndRef.current?.scrollIntoView({ behavior: "smooth" });
      setBotHasResponded(false);
    }
  }, [botHasResponded]);

  const sendMessage = async () => {
    if (!inputValue.trim()) return;
    const userMessage = inputValue;
    setMessages(prev => [...prev, { id: uuidv4(), text: userMessage, sender: 'user' }]);
    setInputValue('');
    await handleBotResponse(userMessage);
  };

  const handleBotResponse = async (userMessage: string) => {
    setIsBotTyping(true);
    try {
      const response = await fetch(`http://localhost:8000/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ human_say: userMessage }),
      });

      if (!response.ok) throw new Error(`Network response was not ok: ${response.statusText}`);

      const data = await response.json();
      setThinkingProcess(prev => [...prev, {
        conversationalStage: data.conversational_stage,
        tool: data.tool,
        toolInput: data.tool_input,
        actionOutput: data.action_output,
        actionInput: data.action_input
      }]);
      setMessages(prev => [...prev, { id: uuidv4(), text: data.response, sender: 'bot' }]);
    } catch (error) {
      console.error("Failed to fetch bot's response:", error);
    } finally {
      setIsBotTyping(false);
      setBotHasResponded(true);
    }
  };

  const renderMessage = (message: Message) => (
    <div key={message.id} className="flex items-center p-2">
      {message.sender === 'user' ? (
        <>
          <span role="img" aria-label="User" className="mr-2">👤</span>
          <span className="text-frame p-2 rounded-lg bg-blue-100 dark:bg-blue-900 text-blue-900">
            {message.text}
          </span>
        </>
      ) : (
        <div className="flex w-full justify-between">
          <div className="flex items-center">
            <img alt="Bot" className="rounded-full mr-2" src="/maskot.png" style={{ width: 24, height: 24, objectFit: "cover" }} />
            <span className="text-frame p-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-900">
              <ReactMarkdown rehypePlugins={[rehypeRaw]} components={{
                a: ({node, ...props}) => <a {...props} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-700" />
              }}>
                {message.text}
              </ReactMarkdown>
            </span>
          </div>
          <div className="flex items-center justify-end ml-2">
            <div className="text-sm text-gray-500" style={{minWidth: '20px', textAlign: 'right'}}>
              <strong>({messages.filter(m => m.sender === 'bot').length})</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderThinkingProcess = (process: ThinkingProcess, index: number) => (
    <div key={index} className="break-words my-2">
      <div><strong>({index + 1})</strong></div>
      <div><strong>Conversational Stage:</strong> {process.conversationalStage}</div>
      {process.tool && <div><strong>Tool:</strong> {process.tool}</div>}
      {process.toolInput && <div><strong>Tool Input:</strong> {process.toolInput}</div>}
      {process.actionInput && <div><strong>Action Input:</strong> {process.actionInput}</div>}
      {process.actionOutput && <div><strong>Action Output:</strong> {process.actionOutput}</div>}
    </div>
  );

  return (
    <div className="flex flex-col" style={{ height: '89vh' }}>
      <header className="flex items-center justify-center h-16 bg-gray-900 text-white">
        <BotIcon className="animate-wave h-7 w-6 mr-2" />
        <h1 className="text-2xl font-bold">LeadGPT</h1>
      </header>
      <main className="flex flex-row justify-center items-start bg-gray-100 dark:bg-gray-900 p-4">
        <div className="flex flex-col w-1/2 h-full bg-white rounded-lg shadow-md p-4 mr-4 chat-messages" style={{maxHeight}}>
          <div className="flex items-center mb-4">
            <BotIcon className="h-6 w-6 text-gray-500 mr-2" />
            <h2 className="text-lg font-semibold">Chat Interface With The Customer</h2>
          </div>
          <div className={`flex-1 overflow-y-auto ${styles.hideScrollbar}`}>
            {messages.map(renderMessage)}
            {isBotTyping && (
              <div className="flex items-center justify-start">
                <img alt="Bot" className="rounded-full mr-2" src="/maskot.png" style={{ width: 24, height: 24, objectFit: "cover" }} />
                <div className={styles.typingBubble}>
                  <span className={styles.typingDot}></span>
                  <span className={styles.typingDot}></span>
                  <span className={styles.typingDot}></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="mt-4">
            <Input
              className="w-full"
              placeholder="Type your message..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            />
          </div>
        </div>
        <div className="flex flex-col w-1/2 h-full bg-white rounded-lg shadow-md p-4 thinking-process" style={{maxHeight}}>
          <div className="flex items-center mb-4">
            <BotIcon className="h-6 w-6 text-gray-500 mr-2" />
            <h2 className="text-lg font-semibold">AI Lead Agent Thought Process</h2>
          </div>
          <div className={`flex-1 overflow-y-auto hide-scroll ${styles.hideScrollbar}`} style={{ overflowX: 'hidden' }}>
            {thinkingProcess.map(renderThinkingProcess)}
            <div ref={thinkingProcessEndRef} />
          </div>
        </div>
      </main>
    </div>
  );
}

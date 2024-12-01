import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Input } from "@/components/ui/input";
import LoaderIcon from '@/components/ui/loader-icon';
import styles from './ChatInterface.module.css';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'bot';
};

type ThinkingProcess = {
  current_stage_id: string;
  current_conversation_stage: string;
  customer_information: string | null;
  thoughts: string[];
  actions: string[];
  action_inputs: string[];
  observations: string[];
  final_thought: string;
  final_response: string;
};

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [thinkingProcess, setThinkingProcess] = useState<ThinkingProcess | null>(null);
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
        body: JSON.stringify({ content: userMessage }),
      });

      if (!response.ok) throw new Error(`Network response was not ok: ${response.statusText}`);

      const data = await response.json();
      console.log('Bot response data:', data);
      if (data && data.response) {
        const parsedResponse = JSON.parse(data.response);
        setThinkingProcess(parsedResponse);
        setMessages(prev => [...prev, { id: uuidv4(), text: parsedResponse.final_response, sender: 'bot' }]);
      } else {
        console.error("Unexpected response format:", data);
        throw new Error("Unexpected response format");
      }
    } catch (error) {
      console.error("Failed to fetch bot's response:", error);
      setMessages(prev => [...prev, { id: uuidv4(), text: "Sorry, there was an error processing your request.", sender: 'bot' }]);
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
            <span role="img" aria-label="Bot" className="mr-2">🌼</span>
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

  const renderThinkingProcess = () => {
    if (!thinkingProcess) return null;
    return (
      <div className="break-words my-2">
        <div><strong>Current Stage ID:</strong> {thinkingProcess.current_stage_id}</div>
        <div><strong>Current Conversation Stage:</strong> {thinkingProcess.current_conversation_stage}</div>
        <div><strong>Customer Information:</strong> {thinkingProcess.customer_information || 'Not available'}</div>
        {thinkingProcess.thoughts && thinkingProcess.thoughts.length > 0 && (
          <div>
            <strong>Thoughts:</strong>
            <ul>
              {thinkingProcess.thoughts.map((thought, index) => (
                <li key={index}>{thought}</li>
              ))}
            </ul>
          </div>
        )}
        {thinkingProcess.actions && thinkingProcess.actions.length > 0 && (
          <div>
            <strong>Actions:</strong>
            <ul>
              {thinkingProcess.actions.map((action, index) => (
                <li key={index}>{action}</li>
              ))}
            </ul>
          </div>
        )}
        {thinkingProcess.action_inputs && thinkingProcess.action_inputs.length > 0 && (
          <div>
            <strong>Action Inputs:</strong>
            <ul>
              {thinkingProcess.action_inputs.map((input, index) => (
                <li key={index}>{input}</li>
              ))}
            </ul>
          </div>
        )}
        {thinkingProcess.observations && thinkingProcess.observations.length > 0 && (
          <div>
            <strong>Observations:</strong>
            <ul>
              {thinkingProcess.observations.map((observation, index) => (
                <li key={index}>{observation}</li>
              ))}
            </ul>
          </div>
        )}
        <div><strong>Final Thought:</strong> {thinkingProcess.final_thought}</div>
        <div><strong>Final Response:</strong> {thinkingProcess.final_response}</div>
      </div>
    );
  };

  return (
    <div className="flex flex-col" style={{ height: '89vh' }}>
      <header className="flex items-center justify-center h-16 bg-gray-900 text-white">
        <span className="text-2xl mr-2"></span>
        <h1 className="text-2xl font-bold">LeadGPT</h1>
      </header>
      <main className="flex flex-row justify-center items-start bg-gray-100 dark:bg-gray-900 p-4">
        <div className="flex flex-col w-1/2 h-full bg-white rounded-lg shadow-md p-4 mr-4 chat-messages" style={{maxHeight}}>
          <div className="flex items-center mb-4">
            <span className="text-xl mr-2">💬</span>
            <h2 className="text-lg font-semibold">Chat Interface With The Customer</h2>
          </div>
          <div className={`flex-1 overflow-y-auto ${styles.hideScrollbar}`}>
            {messages.map(renderMessage)}
            {isBotTyping && (
              <div className="flex items-center justify-start">
                <span role="img" aria-label="Bot" className="mr-2">🌼</span>
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
            <span className="text-xl mr-2">🧠</span>
            <h2 className="text-lg font-semibold">AI Lead Agent Thought Process</h2>
          </div>
          <div className={`flex-1 overflow-y-auto hide-scroll ${styles.hideScrollbar}`} style={{ overflowX: 'hidden' }}>
            {renderThinkingProcess()}
            <div ref={thinkingProcessEndRef} />
          </div>
        </div>
      </main>
    </div>
  );
}
